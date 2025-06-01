#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Environment name - match with setup.sh
ENV_NAME="metanavit"

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        return 1
    fi
    return 0
}

# Function to install conda if not present
install_conda() {
    echo -e "${YELLOW}Installing Miniconda...${NC}"
    
    # Download and install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc
    
    echo -e "${GREEN}Miniconda installed successfully!${NC}"
}

# Function to setup conda environment
setup_conda_env() {
    echo -e "${YELLOW}Setting up conda environment...${NC}"
    
    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        echo -e "${YELLOW}Conda not found. Installing Miniconda...${NC}"
        install_conda
    fi
    
    # Initialize conda for bash if not already done
    if ! conda info --envs &> /dev/null; then
        echo -e "${YELLOW}Initializing conda...${NC}"
        conda init bash
        source ~/.bashrc
    fi
    
    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Environment $ENV_NAME already exists. Removing it...${NC}"
        conda env remove -n $ENV_NAME -y
    fi
    
    # Create new conda environment with Python 3.11
    echo -e "${YELLOW}Creating conda environment '$ENV_NAME' with Python 3.11...${NC}"
    conda create -n $ENV_NAME python=3.11 -y
    
    # Activate the environment
    echo -e "${YELLOW}Activating conda environment...${NC}"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    # Verify activation
    if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
        echo -e "${RED}Failed to activate conda environment${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Conda environment '$ENV_NAME' created and activated!${NC}"
}

# Function to install system dependencies
install_system_deps() {
    echo -e "${YELLOW}Installing system dependencies...${NC}"
    
    # Update package list
    sudo apt-get update
    
    # Install dos2unix and other system dependencies
    sudo apt-get install -y dos2unix curl wget git
    
    echo -e "${GREEN}System dependencies installed successfully!${NC}"
}

# Function to install Python dependencies
install_python_deps() {
    echo -e "${YELLOW}Installing Python dependencies in conda environment...${NC}"
    
    # Make sure we're in the right environment
    if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME
    fi
    
    # Install conda packages first (faster and more reliable)
    echo -e "${YELLOW}Installing conda packages...${NC}"
    conda install -c conda-forge -y \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        jupyter \
        requests \
        psycopg2 \
        sqlalchemy \
        fastapi \
        uvicorn \
        pydantic \
        python-multipart \
        jinja2 \
        aiofiles \
        python-dotenv
    
    # Install pip packages that aren't available in conda
    echo -e "${YELLOW}Installing pip packages...${NC}"
    pip install -U deepeval
    pip install lm-format-enforcer
    
    # Install from requirements.txt if it exists
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    fi
    
    echo -e "${GREEN}Python dependencies installed successfully!${NC}"
}

# Function to install Node.js dependencies
install_node_deps() {
    echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        echo -e "${YELLOW}Installing Node.js via conda...${NC}"
        conda install -c conda-forge nodejs npm -y
    fi
    
    # Install npm dependencies
    if [ -f package.json ]; then
        npm install
    fi
    
    echo -e "${GREEN}Node.js dependencies installed successfully!${NC}"
}

# Function to setup PostgreSQL
setup_postgres() {
    echo -e "${YELLOW}Setting up PostgreSQL...${NC}"
    
    # Check if PostgreSQL is installed
    if ! command -v psql &> /dev/null; then
        echo -e "${YELLOW}Installing PostgreSQL...${NC}"
        sudo apt-get update
        sudo apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all
    fi
    
    # Start PostgreSQL service
    sudo service postgresql start
    
    # Create database and user
    read -p "Enter database name [metanavit]: " DB_NAME
    DB_NAME=${DB_NAME:-metanavit}
    read -p "Enter database user [postgres]: " DB_USER
    DB_USER=${DB_USER:-postgres}
    read -sp "Enter database password: " DB_PASSWORD
    echo  # Add newline after password input
    
    # Check if user already exists and create if not
    if sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1; then
        echo -e "${YELLOW}User $DB_USER already exists, updating password...${NC}"
        sudo -u postgres psql -c "ALTER USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
    else
        echo -e "${YELLOW}Creating user $DB_USER...${NC}"
        sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
    fi
    
    # Grant necessary privileges to user
    sudo -u postgres psql -c "ALTER USER $DB_USER CREATEDB;"
    sudo -u postgres psql -c "ALTER USER $DB_USER WITH SUPERUSER;"
    
    # Check if database exists and create if not
    if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
        echo -e "${YELLOW}Database $DB_NAME already exists...${NC}"
    else
        echo -e "${YELLOW}Creating database $DB_NAME...${NC}"
        sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
    fi
    
    # Install required extensions
    echo -e "${YELLOW}Installing PostgreSQL extensions...${NC}"
    sudo -u postgres psql -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;"
    sudo -u postgres psql -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
    
    # Test the connection
    echo -e "${YELLOW}Testing database connection...${NC}"
    if PGPASSWORD=$DB_PASSWORD psql -h localhost -U $DB_USER -d $DB_NAME -c "SELECT 1;" > /dev/null 2>&1; then
        echo -e "${GREEN}Database connection successful!${NC}"
    else
        echo -e "${RED}Database connection failed. Please check your credentials.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}PostgreSQL setup completed!${NC}"
}

# Function to setup Ollama
setup_ollama() {
    echo -e "${YELLOW}Setting up Ollama...${NC}"
    
    # Check if Ollama is installed
    if ! command -v ollama &> /dev/null; then
        echo -e "${YELLOW}Installing Ollama...${NC}"
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    
    # Start Ollama service
    ollama serve &
    sleep 5  # Give Ollama time to start
    
    # Ask for model preferences
    echo -e "${YELLOW}Available models:${NC}"
    echo "1. llama3.2:1b (1.3 GB, lightweight)"
    echo "2. llama2 (3.8 GB, recommended for general use)"
    echo "3. llama3.2:3b (2.0 GB, balanced)"
    
    read -p "Select LLM model (1-3) [1]: " MODEL_CHOICE
    MODEL_CHOICE=${MODEL_CHOICE:-1}
    
    case $MODEL_CHOICE in
        1) MODEL="llama3.2:1b" ;;
        2) MODEL="llama2" ;;
        3) MODEL="llama3.2:3b" ;;
        *) MODEL="llama3.2:1b" ;;
    esac
    
    # Pull selected model
    echo -e "${YELLOW}Pulling $MODEL model...${NC}"
    ollama pull $MODEL
    
    # Note: We don't pull nomic-embed-text anymore since we're using HuggingFace embeddings
    echo -e "${GREEN}Ollama setup completed!${NC}"
    echo -e "${YELLOW}Note: Using HuggingFace embedding model for better compatibility${NC}"
}

# Function to create .env file
create_env_file() {
    echo -e "${YELLOW}Creating .env file...${NC}"
    
    # Create .env file with all necessary variables
    cat > .env << EOL
# The provider for the AI models to use.
MODEL_PROVIDER=$MODEL_PROVIDER

# The name of LLM model to use.
MODEL=$MODEL
OLLAMA_BASE_URL=http://localhost:11434

# Name of the embedding model to use.
EMBEDDING_MODEL=$EMBEDDING_MODEL

# Dimension of the embedding model to use.
EMBEDDING_DIM=$EMBEDDING_DIM

# The questions to help users get started (multi-line).
# CONVERSATION_STARTERS=

# The OpenAI API key to use.
$([ -n "$OPENAI_API_KEY" ] && echo "OPENAI_API_KEY=$OPENAI_API_KEY" || echo "#OPENAI_API_KEY=")

# The Google API key to use.
$([ -n "$GOOGLE_API_KEY" ] && echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" || echo "#GOOGLE_API_KEY=")

# The Anthropic API key to use.
$([ -n "$ANTHROPIC_API_KEY" ] && echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" || echo "#ANTHROPIC_API_KEY=")

# The Groq API key to use.
$([ -n "$GROQ_API_KEY" ] && echo "GROQ_API_KEY=$GROQ_API_KEY" || echo "#GROQ_API_KEY=")

# The Mistral API key to use.
$([ -n "$MISTRAL_API_KEY" ] && echo "MISTRAL_API_KEY=$MISTRAL_API_KEY" || echo "#MISTRAL_API_KEY=")

# The HuggingFace API key to use.
$([ -n "$HUGGINGFACE_API_KEY" ] && echo "HUGGINGFACE_API_KEY=$HUGGINGFACE_API_KEY" || echo "#HUGGINGFACE_API_KEY=")

# Azure OpenAI configuration.
$([ -n "$AZURE_OPENAI_API_KEY" ] && echo "AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY" || echo "#AZURE_OPENAI_API_KEY=")
$([ -n "$AZURE_OPENAI_ENDPOINT" ] && echo "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" || echo "#AZURE_OPENAI_ENDPOINT=")

# Temperature for sampling from the model.
# LLM_TEMPERATURE=0.2

# Maximum number of tokens to generate.
# LLM_MAX_TOKENS=2048

# The number of similar embeddings to return when retrieving documents.
# TOP_K=5

# PostGreSQL database URL.
PG_CONNECTION_STRING="postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME"
PSYCOPG2_CONNECTION_STRING="dbname=$DB_NAME user=$DB_USER password=$DB_PASSWORD host=localhost port=5432"
DB_NAME="$DB_NAME"

# Storage management
FRONTEND_DIR=".frontend"
STATIC_DIR="static"
DATA_DIR="data/"
STORAGE_DIR="storage"

# The directory to store the local storage cache.
STORAGE_CACHE_DIR=.cache

# FILESERVER_URL_PREFIX is the URL prefix of the server storing the images generated by the interpreter.
FILESERVER_URL_PREFIX=http://localhost:8000/api/files

# The address to start the backend app.
APP_HOST=0.0.0.0

# The port to start the backend app.
APP_PORT=8000

# Customize prompt to generate the next question suggestions based on the conversation history.
# Disable this prompt to disable the next question suggestions feature.
NEXT_QUESTION_PROMPT="You're a helpful assistant! Your task is to suggest the next question that user might ask. 
Here is the conversation history
---------------------
{conversation}
---------------------
Given the conversation history, please give me 3 questions that user might ask next!
Your answer should be wrapped in three sticks which follows the following format:
\`\`\`language=
<question 1>
<question 2>
<question 3>
\`\`\`"

# The system prompt for the AI model.
SYSTEM_PROMPT="You are a helpful AI file system assistant. Your task is to help the user understand and perform tasks related to their files."

# Conda environment name
CONDA_ENV_NAME=$ENV_NAME
EOL

    echo -e "${GREEN}.env file created successfully!${NC}"
}

# Function to create conda activation script
create_activation_script() {
    echo -e "${YELLOW}Creating conda activation script...${NC}"
    
    cat > activate_metanavit.sh << 'EOL'
#!/bin/bash
# MetaNaviT Conda Environment Activation Script

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Activating MetaNaviT conda environment...${NC}"

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the environment
conda activate metanavit

if [[ "$CONDA_DEFAULT_ENV" == "metanavit" ]]; then
    echo -e "${GREEN}MetaNaviT environment activated successfully!${NC}"
    echo -e "${YELLOW}You can now run:${NC}"
    echo -e "  ./scripts/run.sh dev    # Start development server"
    echo -e "  ./scripts/run.sh build  # Build the application"
else
    echo -e "${RED}Failed to activate MetaNaviT environment${NC}"
fi
EOL

    chmod +x activate_metanavit.sh
    echo -e "${GREEN}Activation script created: activate_metanavit.sh${NC}"
}

# Function to create desktop shortcut
create_desktop_shortcut() {
    echo -e "${YELLOW}Creating desktop shortcut...${NC}"
    
    # Create desktop entry
    cat > ~/Desktop/MetaNaviT.desktop << EOL
[Desktop Entry]
Version=1.0
Type=Application
Name=MetaNaviT
Comment=MetaNaviT Application
Exec=bash -c "cd $(pwd) && source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME && ./scripts/run.sh dev"
Icon=$(pwd)/static/icon.png
Terminal=true
Categories=Development;
EOL
    
    # Make it executable
    chmod +x ~/Desktop/MetaNaviT.desktop
    
    echo -e "${GREEN}Desktop shortcut created!${NC}"
}

# Function to setup models
setup_models() {
    echo -e "${YELLOW}Setting up AI Models...${NC}"
    
    # Show available model providers
    echo -e "${YELLOW}Available model providers:${NC}"
    echo "1. ollama (local, no API key needed)"
    echo "2. openai (requires API key)"
    echo "3. anthropic (requires API key)"
    echo "4. groq (requires API key)"
    echo "5. mistral (requires API key)"
    echo "6. gemini (requires API key)"
    echo "7. huggingface (requires API key)"
    echo "8. azure-openai (requires API key)"
    
    read -p "Select model provider (1-8) [1]: " PROVIDER_CHOICE
    PROVIDER_CHOICE=${PROVIDER_CHOICE:-1}
    
    case $PROVIDER_CHOICE in
        1) 
            MODEL_PROVIDER="ollama"
            setup_ollama
            ;;
        2) 
            MODEL_PROVIDER="openai"
            read -p "Enter OpenAI API key: " OPENAI_API_KEY
            MODEL="gpt-4"
            ;;
        3) 
            MODEL_PROVIDER="anthropic"
            read -p "Enter Anthropic API key: " ANTHROPIC_API_KEY
            echo -e "${YELLOW}Available Claude models:${NC}"
            echo "1. claude-3-opus"
            echo "2. claude-3-sonnet"
            echo "3. claude-3-haiku"
            echo "4. claude-2.1"
            read -p "Select Claude model (1-4) [2]: " CLAUDE_CHOICE
            case $CLAUDE_CHOICE in
                1) MODEL="claude-3-opus" ;;
                2) MODEL="claude-3-sonnet" ;;
                3) MODEL="claude-3-haiku" ;;
                4) MODEL="claude-2.1" ;;
                *) MODEL="claude-3-sonnet" ;;
            esac
            ;;
        4) 
            MODEL_PROVIDER="groq"
            read -p "Enter Groq API key: " GROQ_API_KEY
            MODEL="llama2-70b-4096"
            ;;
        5) 
            MODEL_PROVIDER="mistral"
            read -p "Enter Mistral API key: " MISTRAL_API_KEY
            MODEL="mistral-large"
            ;;
        6) 
            MODEL_PROVIDER="gemini"
            read -p "Enter Google API key: " GOOGLE_API_KEY
            MODEL="gemini-pro"
            ;;
        7) 
            MODEL_PROVIDER="huggingface"
            read -p "Enter HuggingFace API key: " HUGGINGFACE_API_KEY
            read -p "Enter model name: " MODEL
            ;;
        8) 
            MODEL_PROVIDER="azure-openai"
            read -p "Enter Azure OpenAI API key: " AZURE_OPENAI_API_KEY
            read -p "Enter Azure OpenAI endpoint: " AZURE_OPENAI_ENDPOINT
            read -p "Enter model name: " MODEL
            ;;
        *) 
            MODEL_PROVIDER="ollama"
            setup_ollama
            ;;
    esac
    
    # Setup embedding model
    setup_embedding_model
}

# Function to get embedding dimensions
get_embedding_dimensions() {
    case $EMBEDDING_MODEL in
        "text-embedding-3-large") echo 3072 ;;
        "text-embedding-3-small") echo 1536 ;;
        "text-embedding-ada-002") echo 1536 ;;
        "nomic-embed-text") echo 768 ;;
        "all-MiniLM-L6-v2") echo 384 ;;
        "all-mpnet-base-v2") echo 768 ;;
        "paraphrase-multilingual-mpnet-base-v2") echo 768 ;;
        *) 
            echo -e "${YELLOW}Unknown embedding model. Please enter dimensions manually:${NC}"
            read -p "Enter embedding dimensions: " EMBEDDING_DIM
            echo $EMBEDDING_DIM
            ;;
    esac
}

# Function to setup embedding model
setup_embedding_model() {
    echo -e "${YELLOW}Setting up embedding model...${NC}"
    
    case $MODEL_PROVIDER in
        "openai")
            echo -e "${YELLOW}Select OpenAI embedding model:${NC}"
            echo "1. text-embedding-3-large (3072 dimensions, best quality)"
            echo "2. text-embedding-3-small (1536 dimensions, faster)"
            echo "3. text-embedding-ada-002 (1536 dimensions, legacy)"
            read -p "Select embedding model (1-3) [1]: " EMBEDDING_CHOICE
            case $EMBEDDING_CHOICE in
                1) EMBEDDING_MODEL="text-embedding-3-large" ;;
                2) EMBEDDING_MODEL="text-embedding-3-small" ;;
                3) EMBEDDING_MODEL="text-embedding-ada-002" ;;
                *) EMBEDDING_MODEL="text-embedding-3-large" ;;
            esac
            ;;
        "azure-openai")
            EMBEDDING_MODEL="text-embedding-3-large"
            ;;
        "ollama")
            # For Ollama, we need to use a HuggingFace model that works
            echo -e "${YELLOW}Select embedding model for Ollama:${NC}"
            echo "1. all-MiniLM-L6-v2 (384 dimensions, lightweight)"
            echo "2. all-mpnet-base-v2 (768 dimensions, balanced)"
            echo "3. paraphrase-multilingual-mpnet-base-v2 (768 dimensions, multilingual)"
            read -p "Select embedding model (1-3) [1]: " EMBEDDING_CHOICE
            case $EMBEDDING_CHOICE in
                1) EMBEDDING_MODEL="all-MiniLM-L6-v2" ;;
                2) EMBEDDING_MODEL="all-mpnet-base-v2" ;;
                3) EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2" ;;
                *) EMBEDDING_MODEL="all-MiniLM-L6-v2" ;;
            esac
            ;;
        *)
            echo -e "${YELLOW}Select embedding model:${NC}"
            echo "1. all-MiniLM-L6-v2 (384 dimensions, lightweight)"
            echo "2. all-mpnet-base-v2 (768 dimensions, balanced)"
            echo "3. paraphrase-multilingual-mpnet-base-v2 (768 dimensions, multilingual)"
            read -p "Select embedding model (1-3) [1]: " EMBEDDING_CHOICE
            case $EMBEDDING_CHOICE in
                1) EMBEDDING_MODEL="all-MiniLM-L6-v2" ;;
                2) EMBEDDING_MODEL="all-mpnet-base-v2" ;;
                3) EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2" ;;
                *) EMBEDDING_MODEL="all-MiniLM-L6-v2" ;;
            esac
            ;;
    esac
    
    # Get embedding dimensions automatically
    EMBEDDING_DIM=$(get_embedding_dimensions)
}

# Main setup function
main() {
    echo -e "${GREEN}Starting MetaNaviT setup with conda environment...${NC}"
    
    # Install system dependencies first
    install_system_deps
    
    # Setup conda environment
    setup_conda_env
    
    # Install Python dependencies in conda environment
    install_python_deps
    
    # Install Node.js dependencies
    install_node_deps
    
    # Setup PostgreSQL
    setup_postgres
    
    # Setup Models
    setup_models
    
    # Create .env file (this must come after setup_postgres and setup_models)
    create_env_file
    
    # Create activation script
    create_activation_script
    
    # Verify .env file was created correctly
    echo -e "${YELLOW}Verifying .env file...${NC}"
    if [ -f .env ]; then
        echo -e "${GREEN}.env file exists${NC}"
        echo -e "${YELLOW}Database configuration:${NC}"
        grep "PG_CONNECTION_STRING\|DB_NAME\|MODEL_PROVIDER\|MODEL\|EMBEDDING_MODEL" .env
    else
        echo -e "${RED}.env file not found!${NC}"
        exit 1
    fi
    
    # Build the application
    echo -e "${YELLOW}Building the application...${NC}"
    ./scripts/run.sh build
    
    # Create desktop shortcut
    create_desktop_shortcut
    
    echo -e "${GREEN}Setup completed successfully!${NC}"
    echo -e "${YELLOW}To use MetaNaviT:${NC}"
    echo -e "1. Activate the conda environment: ${GREEN}source activate_metanavit.sh${NC}"
    echo -e "2. Run the application: ${GREEN}./scripts/run.sh dev${NC}"
    echo -e "3. Or double-click the MetaNaviT icon on your desktop"
    echo -e ""
    echo -e "${YELLOW}Conda environment name: ${GREEN}$ENV_NAME${NC}"
}

# Run the setup
main 
