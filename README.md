# Trusted Data Agent

**A transparent, AI-powered conversational interface for Teradata databases.**

The Trusted Data Agent is a sophisticated web application designed to showcase and facilitate AI-powered interaction with a Teradata database system. Its primary goal is to act as the perfect "study buddy" for developers, data analysts, and architects who are exploring the integration of Large Language Models (LLMs) with enterprise data platforms. It provides complete, real-time transparency into the conversational flow between the user, the AI agent, the Teradata Multi-Capability Platform (MCP) server, and the underlying database.

*(Replace with an actual screenshot of the application)*

---

## Table of Contents
- [Overview](#overview)
- [How It Works: Architecture](#how-it-works-architecture)
- [Key Features](#key-features)
- [Installation and Setup Guide](#installation-and-setup-guide)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Set Up Dependencies](#step-2-set-up-dependencies)
  - [Step 3: Configure API Key](#step-3-configure-api-key)
- [Running the Application](#running-the-application)
- [User Guide](#user-guide)
  - [First-Time Setup: Connecting to MCP](#first-time-setup-connecting-to-mcp)
  - [Navigating the Interface](#navigating-the-interface)
  - [Starting a Conversation](#starting-a-conversation)
- [Troubleshooting](#troubleshooting)
- [Author & Contributions](#author--contributions)

---

## Overview

In a world increasingly driven by data, the ability to interact with complex database systems in a simple, intuitive way is paramount. The Trusted Data Agent bridges this gap by providing a natural language interface for your Teradata system. Instead of writing complex SQL queries, you can simply ask questions. The AI agent then:

1.  **Understands** your request.
2.  **Formulates a plan** to find the answer.
3.  **Uses a suite of tools** (exposed by the MCP Server) to interact with the database.
4.  **Synthesizes the results** into a clear, human-readable response.

What makes this agent unique is its commitment to transparency. The **Live Status** panel shows you every thought, every tool call, and every piece of data the agent uses, making it an unparalleled educational tool for understanding how AI agents operate in a real-world data environment.

## How It Works: Architecture

The application operates on a client-server model, with a clear separation of concerns between the user interface and the backend logic.

```
+-----------+      +-------------------------+      +------------------+      +--------------------+      +------------------+
|           |      |                         |      |                  |      |                    |      |                  |
| End User  | <--> |  Frontend (index.html)  | <--> | Backend (Python) | <--> | Google Gemini LLM  | <--> | Teradata MCP     |
|           |      |     (HTML, JS, CSS)     |      |   (Quart Server) |      | (Reasoning Engine) |      | Server (Tools)   |
|           |      |                         |      |                  |      |                    |      |                  |
+-----------+      +-------------------------+      +------------------+      +--------------------+      +------------------+
```

1.  **Frontend (`index.html`):** A single-page application built with HTML, Tailwind CSS, and vanilla JavaScript. It captures user input and uses Server-Sent Events (SSE) to receive real-time updates from the backend, creating a dynamic and responsive experience.
2.  **Backend (`mcp_web_client.py`):** A powerful asynchronous web server built with **Quart**. It serves the frontend, manages user sessions, and orchestrates the entire AI workflow.
3.  **LLM (Google Gemini):** The reasoning engine of the agent. The backend sends structured prompts (including conversation history and available tools) to the Gemini API, which decides the next best action.
4.  **Teradata MCP Server:** The bridge to the database. It exposes database functionalities (like listing tables, describing columns, checking data quality) as a secure, well-defined API of "tools" that the AI agent can call.

## Key Features

* **Intuitive Conversational UI:** Ask questions in plain English to query and analyze your database.
* **Complete Transparency:** The **Live Status** panel provides a real-time, step-by-step stream of the agent's thought process, tool selections, and API results.
* **Dynamic Capability Loading:** Automatically discovers and displays all available **Tools**, **Prompts**, and **Resources** from the connected MCP Server.
* **Rich Data Rendering:** Intelligently formats and displays various data types, including query results in interactive tables and SQL DDL in highlighted code blocks.
* **Persistent Session History:** Keeps a record of your conversations, allowing you to switch between different lines of inquiry.
* **Interactive Workspace:** Features collapsible side and top panels to customize your view and focus on what matters.
* **Easy Configuration:** A simple startup modal allows you to configure the connection to your MCP Server.

## Installation and Setup Guide

Follow these instructions to get the Trusted Data Agent running on your local machine.

### Prerequisites

Before you begin, ensure you have the following:

* **Python 3.8+** and `pip`.
* Access to a running **Teradata MCP Server**. You will need its host, port, and path.
* A **Google Gemini API Key**. You can obtain one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

### Step 1: Clone the Repository

The Trusted Data Agent is available on GitHub. Clone the repository to your local machine:

```bash
git clone [https://github.com/rgeissen/teradata-trusted-data-agent.git](https://github.com/rgeissen/teradata-trusted-data-agent.git)
cd teradata-trusted-data-agent
```

### Step 2: Set Up Dependencies

It is highly recommended to use a Python virtual environment to manage dependencies. This project includes a `requirements.txt` file to simplify the process.

1.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install the required packages from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Configure API Key

The application loads your Gemini API key from a `.env` file for security.

1.  Create a file named `.env` in the root of the project directory.
2.  Add your API key to this file:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```
    Replace `YOUR_GEMINI_API_KEY_HERE` with your actual key.

## Running the Application

With the setup complete, you can now start the backend server.

1.  **Run the Python script:**
    ```bash
    python mcp_web_client.py
    ```

2.  **Access the UI:**
    Once the server is running, you will see a confirmation in your terminal. Open your web browser and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

## User Guide

### First-Time Setup: Connecting to MCP

The first time you launch the application, a configuration modal will appear.
1.  **Host:** Enter the IP address or hostname of your MCP Server (e.g., `127.0.0.1`).
2.  **Port:** Enter the port the MCP Server is running on (e.g., `8001`).
3.  **Path:** Enter the base path for the MCP API (e.g., `/mcp/`).
4.  Click **"Load and Test Connection"**. The application will attempt to connect and load the available capabilities. If successful, the modal will close, and the chat input will become active.

### Navigating the Interface

The UI is divided into several key areas:
* **Capabilities Panel (Top):** Browse available Tools, Prompts, and Resources. Clicking on a prompt allows you to run it directly with arguments.
* **Chat Window (Center):** The main area for your conversation with the agent.
* **Live Status Panel (Right):** A real-time log of the agent's internal state and actions. This is crucial for understanding *how* the agent arrives at its answers.
* **History Panel (Left):** A list of your past and current conversations. Click "New Chat" to start a fresh session.

### Starting a Conversation

Simply type your question into the input box at the bottom and press Enter. Try starting with simple requests and gradually increase complexity.

**Example Prompts:**
* *"List all tables in the `DEMO_Customer360_db` database."*
* *"What is the business description for the `equipment` table?"*
* *"Show me a preview of that table."*
* *"Now, can you check the data quality for the `equipment_id` column?"*

## Troubleshooting

* **Connection Error on Startup:** If the configuration modal shows an error, double-check that your MCP Server is running and that the Host, Port, and Path are correct. Check for firewall issues that might be blocking the connection.
* **LLM Errors:** If you see errors related to the language model, ensure your `.env` file is correctly formatted and contains a valid Gemini API key.
* **"No Tools Available":** This indicates the backend connected to the MCP Server, but the server itself reported having no available tools. Check your MCP Server configuration.

## Author & Contributions

* **Author/Initiator:** Rainer Geissendoerfer, World Wide Data Architecture, Teradata.
* **Source Code & Contributions:** The Trusted Data Agent is open source, and contributions are highly welcome. Please visit the main Git repository to report issues or submit pull requests.
    * **Git Repository:** [https://github.com/rgeissen/teradata-trusted-data-agent.git](https://github.com/rgeissen/teradata-trusted-data-agent.git)
