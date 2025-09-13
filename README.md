# AI-Powered CI Test Runner

An intelligent CI/CD tool that automatically generates and runs unit tests for changed Python files using free LLM APIs.

## Features

- 🤖 **AI-Generated Tests**: Uses free LLM APIs to generate comprehensive unit tests
- 📊 **Code Coverage**: Tracks test coverage with detailed reports
- 🔍 **Smart Analysis**: Analyzes only changed files in PRs
- 🆓 **Free LLM Support**: Works with multiple free LLM providers
- ⚡ **GitHub Actions Ready**: Easy integration with CI/CD pipelines

## Supported LLM Providers

| Provider | Free Tier | Speed | Quality | Setup |
|----------|-----------|-------|---------|-------|
| **Groq** | ✅ High limits | ⚡ Very Fast | 🟢 Good | Easy |
| **OpenRouter** | ✅ Available | 🟡 Medium | 🟢 Good | Easy |
| **Together AI** | ✅ Credits | 🟡 Medium | 🟢 Good | Medium |
| **Hugging Face** | ✅ Available | 🟠 Slower | 🟡 Variable | Easy |

## Setup Instructions

### 1. Choose an LLM Provider

**Recommended: Groq (Fast & Free)**
1. Go to https://console.groq.com/
2. Sign up for a free account
3. Generate an API key
4. Add it to GitHub Secrets as `GROQ_API_KEY`

### 2. Add API Key to GitHub Secrets

1. In your GitHub repo, go to **Settings** → **Secrets and variables** → **Actions**
2. Click **"New repository secret"**
3. Add your chosen secret:
   - `GROQ_API_KEY` for Groq
   - `OPENROUTER_API_KEY` for OpenRouter  
   - `TOGETHER_API_KEY` for Together AI
   - `HUGGINGFACE_API_KEY` for Hugging Face

### 3. Configure the Workflow

The workflow is already configured in `.github/workflows/test-runner.yml`. 

Update the `LLM_PROVIDER` and corresponding API key in the workflow file:

```yaml
env:
  LLM_PROVIDER: "groq"  # Change this to your chosen provider
  GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}  # Match your secret name
```

### 4. Test Locally (Optional)

```bash
# Set your API key
export GROQ_API_KEY="your-api-key-here"
export LLM_PROVIDER="groq"

# Test with specific files
export CHANGED_FILES="main.py utils.py"
python ci_pr_test_runner.py
```

Or use the provided test script:
```bash
./test-local.sh
```

## How It Works

1. **Detects Changes**: Identifies modified Python files in PRs
2. **Analyzes Code**: Extracts functions, methods, and classes using AST
3. **Generates Tests**: Uses AI to create comprehensive unit tests
4. **Runs Tests**: Executes tests with coverage analysis
5. **Reports Results**: Shows test results and coverage metrics

## Example Output

```
🔍 Analyzing changed file: utils.py
🤖 Found 3 functions/methods to test.
✅ Generated and saved tests to tests_pr/pr_generated_tests.py
🧪 Executing tests from tests_pr/pr_generated_tests.py
✅ Tests passed.

📊 Code Coverage Report:
Name      Stmts   Miss  Cover
-----------------------------
utils.py      8      1    88%
-----------------------------
TOTAL         8      1    88%

🎯 Total Code Coverage: 88%
🟡 Good coverage, consider adding more tests
```

## Provider-Specific Notes

### Groq (Recommended)
- **Pros**: Very fast, generous free tier, high quality
- **Cons**: Newer service
- **Setup**: https://console.groq.com/

### OpenRouter  
- **Pros**: Multiple model options, reliable
- **Cons**: Limited free usage
- **Setup**: https://openrouter.ai/

### Together AI
- **Pros**: Good performance, multiple models
- **Cons**: Limited free credits
- **Setup**: https://api.together.xyz/

### Hugging Face
- **Pros**: Many models, established platform
- **Cons**: Can be slower, variable quality
- **Setup**: https://huggingface.co/settings/tokens

## Troubleshooting

**No API Key Error:**
- Ensure you've added the correct secret name to GitHub
- Check the workflow uses the right environment variable

**Tests Not Generated:**
- Verify the LLM provider is supported
- Check API key has sufficient quota
- Review the workflow logs for errors

**Import Errors:**
- Ensure all dependencies are in `requirements.txt`
- Check Python path configuration

## Contributing

Feel free to add support for more LLM providers or improve the test generation logic!
