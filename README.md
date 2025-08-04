# Trusted Data Agent for Teradata

**An Advanced, Dynamic AI Conversational Interface for Enterprise Data Platforms.**

The Trusted Data Agent represents a paradigm shift in how developers, analysts, and architects interact with complex data ecosystems. It is a sophisticated web application designed not only to showcase AI-powered interaction with a Teradata database but to serve as a powerful, fully transparent "study buddy" for mastering the integration of Large Language Models (LLMs) with enterprise data.

This solution provides unparalleled, real-time insight into the complete conversational flow between the user, the AI agent, the Teradata **Model Context Protocol (MCP)** server, and the underlying database, establishing a new standard for clarity and control in AI-driven data analytics.

## Table of Contents

* [Overview: A Superior Approach](#overview-a-superior-approach)
* [How It Works: Architecture](#how-it-works-architecture)
* [Key Features](#key-features)
* [Deterministic vs. Non-Deterministic Execution: A Best Practice](#deterministic-vs-non-deterministic-execution-a-best-practice)
* [Installation and Setup Guide](#installation-and-setup-guide)
  * [Prerequisites](#prerequisites)
  * [Step 1: Clone the Repository](#step-1-clone-the-repository)
  * [Step 2: Set Up the Python Environment](#step-2-set-up-the-python-environment)
  * [Step 3: Create the Project Configuration File](#step-3-create-the-project-configuration-file)
  * [Step 4: Install the Application in Editable Mode](#step-4-install-the-application-in-editable-mode)
  * [Step 5: Configure API Keys (Optional)](#step-5-configure-api-keys-optional)
* [Running the Application](#running-the-application)
  * [Standard Mode](#standard-mode)
  * [Developer Mode: Unlocking Models](#developer-mode-unlocking-models)
  * [Developer Mode: Enabling Charting](#developer-mode-enabling-charting)
* [User Guide](#user-guide)
* [Troubleshooting](#troubleshooting)
* [License](#license)
* [Author & Contributions](#author--contributions)

## Overview: A Superior Approach

The Trusted Data Agent transcends typical data chat applications by placing ultimate control and understanding in the hands of the user. It provides a seamless natural language interface to your Teradata system, empowering you to ask complex questions and receive synthesized, accurate answers without writing a single line of SQL.

Its core superiority lies in its **unmatched transparency and dynamic configurability**:

1. **Deep Insight:** The **Live Status** panel is more than a log; it's a real-time window into the AI's mind, revealing its reasoning, tool selection, and the raw data it receives. This makes it an indispensable tool for debugging, learning, and building trust in AI systems.

2. **Unprecedented Flexibility:** Unlike static applications, the Trusted Data Agent allows you to dynamically configure your LLM provider, select specific models, and even edit the core **System Prompt** that dictates the agent's behavior—all from within the UI.

3. **Comparative LLM Analysis:** The ability to instantly switch between different LLM providers (e.g., Google, Anthropic, and AWS Bedrock) and their models is a critical feature for developers. It allows for direct, real-time testing of how different reasoning engines interpret the same MCP tools and prompts. This is invaluable for validating the robustness of MCP capabilities and understanding the nuances of various LLMs in an enterprise context.

4. **Intelligent Workflow Routing:** The agent intelligently distinguishes between novel requests requiring creative LLM reasoning and structured, multi-step tasks. For the latter, it switches to a faster, more reliable, and token-efficient deterministic execution algorithm, enhancing stability and performance.

## How It Works: Architecture

The application operates on a sophisticated client-server model, ensuring a clean separation of concerns and robust performance.

```
+-----------+      +-------------------------+      +------------------+      +----------------------+      +------------------+
|           |      |                         |      |                  |      |                      |      |                  |
| End User  | <--> |  Frontend (index.html)  | <--> | Backend (Python) | <--> | Large Language Model | <--> | Teradata MCP     |
|           |      |     (HTML, JS, CSS)     |      |   (Quart Server) |      |  (Reasoning Engine)  |      | Server (Tools)   |
|           |      |                         |      |                  |      |                      |      |                  |
+-----------+      +-------------------------+      +------------------+      +----------------------+      +------------------+
```

1. **Frontend (`templates/index.html`):** A sleek, single-page application built with HTML, Tailwind CSS, and vanilla JavaScript. It captures user input and uses Server-Sent Events (SSE) to render real-time updates from the backend.

2. **Backend** (`src/trusted_data_agent/`): A high-performance asynchronous web server built with **Quart**. It serves the frontend, manages user sessions, and orchestrates the entire AI workflow.

3. **Large Language Model (LLM):** The reasoning engine. The backend dynamically initializes the connection to the selected LLM provider (e.g., Google, Anthropic, AWS Bedrock) based on user-provided credentials and sends structured prompts to the model's API.

4. **Teradata** MCP **Server:** The **Model Context Protocol (MCP)** server acts as the secure, powerful bridge to the database, exposing functionalities as a well-defined API of "tools" for the AI agent.

### Code Structure

The Python source code is organized in a standard `src` layout for better maintainability and scalability.

```
/teradata-trusted-data-agent/
|
├── src/
|   └── trusted_data_agent/   # Main Python package
|       ├── api/              # Quart web routes
|       ├── agent/            # Core agent logic (Executor, Formatter)
|       ├── llm/              # LLM provider interaction
|       ├── mcp/              # MCP server interaction
|       ├── core/             # Config, session management, utils
|       └── main.py           # Application entry point
|
├── templates/
|   └── index.html            # Frontend UI
|
├── pyproject.toml              # Project definition
├── requirements.txt
└── ...
```

This structure separates concerns, making it easier to navigate and extend the application's functionality.

## Key Features

* **Multi-Provider LLM Configuration:** Dynamically switch between LLM providers like Google, Anthropic, AWS Bedrock, and **Ollama**, configure API keys or hosts, and select from a list of available models directly within the application's UI.

* **Support for AWS Bedrock:**

  * **Foundation Models:** Directly access and utilize foundational models available on Bedrock.

  * **Inference Profiles:** Connect to custom, provisioned, or third-party models via Bedrock Inference Profiles.

* **Ollama (Local LLM) Integration:** Run the agent with open-source models on your local machine for privacy, offline use, and development.

* **Direct Model Chat:** A dedicated chat interface (accessible from the "Chat" menu item) allows for direct, tool-less conversations with the configured LLM. This is an invaluable feature for testing a model's baseline reasoning capabilities without the complexity of the agent's tool-use logic.

* **Comparative MCP Testing:** The multi-provider support is crucial for testing and validating how different LLMs interpret and utilize the same set of MCP tools and prompts, providing essential insights into model behavior and capability robustness.

* **Live Model Refresh:** Fetch an up-to-date list of supported models from your provider with the click of a button.

* **System Prompt Editor:** Take full control of the agent's behavior. Edit, save, and reset the core system prompt for each model, with changes persisting across sessions.

* **Intuitive Conversational UI:** Ask questions in plain English to query and analyze your database.

* **Complete Transparency:** The **Live Status** panel provides a real-time stream of the agent's thought process, actions, and tool outputs.

* **Dynamic Capability Loading:** Automatically discovers and displays all available **Tools**, **Prompts**, and **Resources** from the connected MCP Server.

* **Rich Data Rendering:** Intelligently formats and displays various data types, including query results in interactive tables and SQL DDL in highlighted code blocks.

* **Optional Charting Engine:** Enable data visualization capabilities to render charts based on query results by using a runtime parameter.

* **Token Usage Tracking:** The application tracks and displays the number of input and output tokens used for each LLM call, providing clear insight into the cost and efficiency of each interaction.

* **Intelligent Workflow Routing (Deterministic vs. Non-Deterministic):** The agent can distinguish between open-ended user requests that require non-deterministic, LLM-driven reasoning and structured tasks that can be executed via a faster, more reliable, and token-efficient deterministic algorithm.

## Deterministic vs. Non-Deterministic Execution: A Best Practice

This agent implements a powerful, best-practice design pattern: the ability to dynamically choose between **non-deterministic (LLM-driven)** and **deterministic (algorithmic)** execution paths. This intelligent routing is key to building robust, efficient, and reliable AI agents.

### 1. Overview

* **Non-Deterministic (LLM-driven) Path:** This is the standard "ReAct" (Reason and Act) loop. The LLM is given a goal and a set of tools. It *reasons* about the next best step, selects a tool, receives the output, and repeats the cycle until it has enough information to answer the user's request. This path is essential for handling novel, ambiguous, or complex queries that require creative problem-solving.

* **Deterministic (Algorithmic) Path:** When the agent recognizes a user's request maps to a predefined, multi-step task (a "workflow"), it bypasses the LLM's step-by-step reasoning. Instead, it executes a hard-coded sequence of tool calls algorithmically. This is like a traditional program running a script. It's predictable, fast, and reliable.

### 2. Positive Impact

Switching to a deterministic path for known workflows has significant advantages:

* **Massive Token Savings:** The biggest benefit is cost and performance. A deterministic workflow makes a handful of direct tool calls. A non-deterministic approach requires multiple "thought" loops with the LLM, where the entire conversation history and tool outputs are sent back and forth. This can consume thousands of tokens for a task that an algorithm can complete with zero additional LLM calls after the initial recognition.

* **Enhanced Stability and Reliability:** LLMs can be unpredictable. They might hallucinate tool calls, get stuck in loops, or fail to follow instructions perfectly. A deterministic algorithm, by contrast, is 100% reliable and will execute the same way every time.

* **Improved Performance with Weaker Models:** This stability is especially critical when using smaller, less powerful models (e.g., via Ollama with on-premise models). These models may struggle with complex, multi-step reasoning. Offloading structured tasks to a deterministic path ensures that even these models can perform complex workflows reliably, making the agent more versatile and accessible.

* **Predictable Execution Time:** Algorithmic execution is significantly faster than waiting for multiple LLM API calls, leading to a much better user experience.

### 3. Implementation Approach

The agent's `PlanExecutor` implements this logic as follows:

1. **Prompt Recognition:** The LLM's first task is to map the user's request to either a one-off tool call or a "workflow prompt."

2. **Workflow Detection:** When the `PlanExecutor` receives a command to execute a prompt, it inspects the prompt's text. If it contains specific keywords (e.g., "phase 1," "cycle through"), it flags it as a deterministic workflow.

3. **Algorithmic Execution:** Instead of feeding the prompt's steps to the LLM one by one, the `PlanExecutor`'s `_parse_and_execute_workflow` function takes over. It parses the tool sequence directly from the prompt text and executes the tool calls in a hard-coded loop, gathering all the data.

4. **Final Summarization:** Once the deterministic data gathering is complete, all the collected results are passed back to the LLM *one final time* for a comprehensive summary.

### 4. Prerequisites for a Deterministic Workflow

For the agent to recognize and execute a deterministic workflow, the corresponding prompt on the MCP server must be structured in a specific way:

* **Keyword Trigger:** The prompt's text **must** contain keywords that signal a multi-phase process. The current implementation looks for phrases like **`## Phase 1`** and **`cycle through`**.

* **Clear Tool Sequence:** The prompt should explicitly name the tools to be used in sequence (e.g., "...first by using the `base_tableList` tool, and then for each table, cycle through `qlty_columnSummary` and `qlty_univariateStatistics`..."). The executor parses these tool names to build its algorithmic plan.

By adhering to this convention, new complex workflows can be added to the MCP server, and the agent will automatically be able to execute them deterministically without any changes to the agent's core code.

## Installation and Setup Guide

### Prerequisites

* **Python 3.8+** and `pip`.

* Access to a running **Teradata MCP Server**.

* An **API** Key from **a supported LLM provider** or a **local Ollama installation**. The initial validated providers are **Google**, **Anthropic**, **Amazon Web Services (AWS)**, and **Ollama**.

  * You can obtain a Gemini API key from the [Google AI Studio](https://aistudio.google.com/app/apikey).

  * You can obtain a Claude API key from the [Anthropic Console](https://console.anthropic.com/dashboard).

  * For AWS, you will need an **AWS Access Key ID**, **Secret Access Key**, and the **Region** for your Bedrock service.

  * For Ollama, download and install it from [ollama.com](https://ollama.com/) and pull a model (e.g., `ollama run llama2`).

### Step 1: Clone the Repository

```
git clone [https://github.com/rgeissen/teradata-trusted-data-agent.git](https://github.com/rgeissen/teradata-trusted-data-agent.git)
cd teradata-trusted-data-agent
```

### Step 2: Set Up the Python Environment

It is highly recommended to use a Python virtual environment.

1. **Create and activate a virtual environment:**

```
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

2. **Install the required packages:**

```
pip install -r requirements.txt
```

### Step 3: Create the Project Configuration File

In the project's root directory, create a new file named `pyproject.toml`. This file is essential for Python to recognize the project structure.

Copy and paste the following content into `pyproject.toml`:

```
[project]
name = "trusted-data-agent"
version = "0.1.0"
requires-python = ">=3.8"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

### Step 4: Install the Application in Editable Mode

This crucial step links your source code to your Python environment, resolving all import paths. **Run this command from the project's root directory.**

```
pip install -e .
```

The `-e` flag stands for "editable," meaning any changes you make to the source code will be immediately effective without needing to reinstall.

### Step 5: Configure API Keys (Optional)

You can either enter your API keys in the UI at runtime or, for convenience during development, create a `.env` file in the project root. The application will automatically load these keys.

```
# For Google Models
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

# For Anthropic Models
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY_HERE"

# For AWS Bedrock Models
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
AWS_REGION="your-bedrock-region"

# For Ollama (Local)
OLLAMA_HOST="http://localhost:11434"
```

## Running the Application

**Important:** All commands must be run from the project's **root directory**.

### Standard Mode

For standard operation with the certified models:

```
python -m trusted_data_agent.main
```

### Developer Mode: Unlocking Models

To enable all discovered models for testing and development purposes, start the server with the `--all-models` flag.

```
python -m trusted_data_agent.main --all-models
```

**Note:** **No Ollama models are currently certified.** For testing purposes, Ollama models can be evaluated by starting the server with the `--all-models` developer flag.

### Developer Mode: Enabling Charting

To enable data visualization capabilities, start the server with the `--charting` flag.

```
python -m trusted_data_agent.main --charting
```

You can also combine flags for a full development environment:

```
python -m trusted_data_agent.main --all-models --charting
```

## User Guide

*(This* section remains the same *as the original)*

## Troubleshooting

* **`ModuleNotFoundError`:** This error almost always means you are either (1) not in the project's root directory, or (2) you have not run `pip install -e .` successfully in your active virtual environment.

* **Connection Errors:** Double-check all host, port, path, and API key information. Ensure no firewalls are blocking the connection. If you receive an API key error, verify that the key is correct and has permissions for the model you selected.

* **"Failed to fetch models":** This usually indicates an invalid API key, an incorrect Ollama host, or a network issue preventing connection to the provider's API.

* **AWS Bedrock Errors:**

  * Ensure your AWS credentials have the necessary IAM permissions (`bedrock:ListFoundationModels`, `bedrock:ListInferenceProfiles`, `bedrock-runtime:InvokeModel`).

  * Verify that the selected model is enabled for access in the AWS Bedrock console for your specified region.

## License

This project is licensed under the GNU Affero General Public License v3.0. The full license text is available in the `LICENSE` file in the root of this repository.

Under the AGPLv3, you are free to use, modify, and distribute this software. However, if you run a modified version of this software on a network server and allow other users to interact with it, you must also make the source code of your modified version available to those users.

## Author & Contributions

* **Author/Initiator:** Rainer Geissendoerfer, World Wide Data Architecture, Teradata.

* **Source** Code & **Contributions:** The Trusted Data Agent is licensed under the GNU Affero General Public License v3.0. Contributions are highly welcome. Please visit the main Git repository to report issues or submit pull requests.

* **Git Repository:** https://github.com/rgeissen/teradata-trusted-data-agent.git
