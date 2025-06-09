package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"path/filepath"

	"github.com/charmbracelet/huh/spinner"
	"github.com/charmbracelet/log"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcphost/pkg/history"
	"github.com/mark3labs/mcphost/pkg/llm"
	"github.com/mark3labs/mcphost/pkg/llm/anthropic"
	"github.com/mark3labs/mcphost/pkg/llm/google"
	"github.com/mark3labs/mcphost/pkg/llm/ollama"
	"github.com/mark3labs/mcphost/pkg/llm/openai"
)

type MCPSession struct {
	History      *history.HistoryMessage
	AllTools     []llm.Tool
	MCPClients   map[string]mcpclient.MCPClient
	MCPServers   map[string]ServerConfigWrapper
	Model        string
	Provider     llm.Provider
	SystemPrompt string
	Config       *MCPConfig
	Verbose      bool
	InTerminal   bool
	DebugMode    bool
}

type InitConfig struct {
	AnthropicAPIKey  string `json:"anthropicApiKey"`
	AnthropicBaseURL string `json:"anthropicBaseUrl"`
	OpenAIAPIKey     string `json:"openaiApiKey"`
	OpenAIBaseURL    string `json:"openaiBaseUrl"`
	GoogleAPIKey     string `json:"googleApiKey"`
	ModelFlag        string `json:"modelFlag"`
	SystemPromptFile string `json:"systemPromptFile"`
	DebugMode        bool   `json:"debugMode"`
	ConfigFile       string `json:"configFile"`
	InTerminal       bool   `json:"inTerminal"`
}

func (ms *MCPSession) LoadMCPConfig(configPath string) error {
	if configPath == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("error getting home directory: %w", err)
		}
		configPath = filepath.Join(homeDir, ".mcp.json")
	}

	// Check if config file exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		// Create default config
		defaultConfig := MCPConfig{
			MCPServers: make(map[string]ServerConfigWrapper),
		}

		// Create the file with default config
		configData, err := json.MarshalIndent(defaultConfig, "", "  ")
		if err != nil {
			return fmt.Errorf("error creating default config: %w", err)
		}

		if err := os.WriteFile(configPath, configData, 0644); err != nil {
			return fmt.Errorf("error writing default config file: %w", err)
		}

		log.Info("Created default config file", "path", configPath)
		return nil
	}

	// Read existing config
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf(
			"error reading config file %s: %w",
			configPath,
			err,
		)
	}

	if err := json.Unmarshal(configData, &ms.Config); err != nil {
		return fmt.Errorf("error parsing config file: %w", err)
	}

	return nil
}

// close all client connections and clean up resources
func (ms *MCPSession) Close() error {
	var err error
	for name, client := range ms.MCPClients {
		if closeErr := client.Close(); closeErr != nil {
			log.Error("Failed to close server", "name", name, "error", closeErr)
			err = closeErr
		} else {
			log.Info("Server closed", "name", name)
		}
	}
	return err
}

// loadSystemPrompt loads the system prompt from a JSON file
func (ms *MCPSession) LoadSystemPrompt(filePath string) error {
	if filePath == "" {
		return nil
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("error reading config file: %v", err)
	}

	// Parse only the systemPrompt field
	var config struct {
		SystemPrompt string `json:"systemPrompt"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("error parsing config file: %v", err)
	}

	ms.SystemPrompt = config.SystemPrompt

	return nil
}

func (ms *MCPSession) CreateMessage(ctx context.Context, prompt string, messages *[]history.HistoryMessage) (llm.Message, error) {
	var message llm.Message
	var err error
	backoff := initialBackoff
	retries := 0

	// Convert MessageParam to llm.Message for provider
	// Messages already implement llm.Message interface
	llmMessages := make([]llm.Message, len(*messages))
	for i := range *messages {
		llmMessages[i] = &(*messages)[i]
	}

	for {
		action := func() {
			message, err = ms.Provider.CreateMessage(
				ctx,
				prompt,
				llmMessages,
				ms.AllTools,
			)
		}
		if ms.InTerminal {
			// If we're in a terminal, use a spinner to indicate processing
			_ = spinner.New().Title("Thinking...").Action(action).Run()
		} else {
			// If we're not in a terminal, just run the action directly
			action()
		}

		if err != nil {
			// Check if it's an overloaded error
			if strings.Contains(err.Error(), "overloaded_error") {
				if retries >= maxRetries {
					return nil, fmt.Errorf(
						"claude is currently overloaded. please wait a few minutes and try again",
					)
				}

				log.Warn("Claude is overloaded, backing off...",
					"attempt", retries+1,
					"backoff", backoff.String())

				time.Sleep(backoff)
				backoff *= 2
				if backoff > maxBackoff {
					backoff = maxBackoff
				}
				retries++
				continue
			}
			// If it's not an overloaded error, return the error immediately
			return nil, err
		}
		// If we got here, the request succeeded
		break
	}
	return message, nil
}

func NewSession(ctx context.Context, cfg InitConfig) (*MCPSession, error) {
	// Set up logging based on debug flag
	if cfg.DebugMode {
		log.SetLevel(log.DebugLevel)
		// Enable caller information for debug logs
		log.SetReportCaller(true)
	} else {
		log.SetLevel(log.InfoLevel)
		log.SetReportCaller(false)
	}

	ms := &MCPSession{
		MCPServers: make(map[string]ServerConfigWrapper),
		Model:      cfg.ModelFlag,
		Config:     &MCPConfig{},
	}

	err := ms.LoadSystemPrompt(cfg.SystemPromptFile)
	if err != nil {
		return nil, fmt.Errorf("error loading system prompt: %v", err)
	}

	// Create the provider based on the model flag
	err = ms.CreateProvider(ctx)
	if err != nil {
		return nil, fmt.Errorf("error creating provider: %v", err)
	}

	// Validate model flag format
	parts := strings.SplitN(cfg.ModelFlag, ":", 2)
	if len(parts) < 2 {
		return nil, fmt.Errorf(
			"invalid model format. Expected provider:model, got %s",
			cfg.ModelFlag,
		)
	}
	log.Info("Model loaded",
		"provider", ms.Provider.Name(),
		"model", parts[1])

	err = ms.LoadMCPConfig(cfg.ConfigFile)
	if err != nil {
		return nil, fmt.Errorf("error loading MCP config: %v", err)
	}

	ms.MCPClients, err = createMCPClients(ms.Config)
	if err != nil {
		return nil, fmt.Errorf("error creating MCP clients: %v", err)
	}

	for name := range ms.MCPClients {
		log.Info("Server connected", "name", name)
	}

	for serverName, mcpClient := range ms.MCPClients {
		ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		toolsResult, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
		cancel()

		if err != nil {
			log.Error(
				"Error fetching tools",
				"server",
				serverName,
				"error",
				err,
			)
			continue
		}

		serverTools := mcpToolsToAnthropicTools(serverName, toolsResult.Tools)
		ms.AllTools = append(ms.AllTools, serverTools...)
		log.Info(
			"Tools loaded",
			"server",
			serverName,
			"count",
			len(toolsResult.Tools),
		)
	}

	// messages := make([]history.HistoryMessage, 0)
	return ms, nil
}

// Add new function to create provider
func (ms *MCPSession) CreateProvider(ctx context.Context) error {
	if ms.SystemPrompt == "" {
		return fmt.Errorf("system prompt is not set")
	}
	if ms.Model == "" {
		return fmt.Errorf("model is not set")
	}
	parts := strings.SplitN(ms.Model, ":", 2)
	if len(parts) < 2 {
		return fmt.Errorf(
			"invalid model format. Expected provider:model, got %s",
			ms.Model,
		)
	}

	provider := parts[0]
	model := parts[1]
	var err error

	switch provider {
	case "anthropic":
		apiKey := anthropicAPIKey
		if apiKey == "" {
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
		}

		if apiKey == "" {
			return fmt.Errorf(
				"Anthropic API key not provided. Use --anthropic-api-key flag or ANTHROPIC_API_KEY environment variable",
			)
		}
		ms.Provider = anthropic.NewProvider(apiKey, anthropicBaseURL, model, ms.SystemPrompt)
		return err

	case "ollama":
		ms.Provider, err = ollama.NewProvider(model, ms.SystemPrompt)
		return err

	case "openai":
		apiKey := openaiAPIKey
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}

		if apiKey == "" {
			return fmt.Errorf(
				"OpenAI API key not provided. Use --openai-api-key flag or OPENAI_API_KEY environment variable",
			)
		}
		ms.Provider = openai.NewProvider(apiKey, openaiBaseURL, model, ms.SystemPrompt)
		return nil

	case "google":
		apiKey := googleAPIKey
		if apiKey == "" {
			apiKey = os.Getenv("GOOGLE_API_KEY")
		}
		if apiKey == "" {
			// The project structure is provider specific, but Google calls this GEMINI_API_KEY in e.g. AI Studio. Support both.
			apiKey = os.Getenv("GEMINI_API_KEY")
		}
		ms.Provider, err = google.NewProvider(ctx, apiKey, model, ms.SystemPrompt)
		return err

	default:
		return fmt.Errorf("unsupported provider: %s", provider)
	}
}
