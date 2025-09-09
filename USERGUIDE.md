# Simtune Stage 1 POC - User Guide

This guide will help you test the core functionality of Simtune's Stage 1 Proof of Concept. You can create a personal AI author that writes in your style through a simple 4-step process.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **OpenAI API key** (required for fine-tuning)
3. **Terminal/Command line** access

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Replace 'your_openai_api_key_here' with your actual API key
```

Your `.env` file should look like:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_ORG_ID=your_openai_org_id_here_optional
```

## Testing the Complete Workflow

### Step 1: Create Your First Author

Start with the init command for a guided setup:

```bash
python -m cli.main init
```

This will walk you through creating an author profile with:

- Author name and description
- Writing style preferences (tone, voice, formality)
- Topics you write about
- Style notes

**Alternative**: Create an author directly:

```bash
python -m cli.main author create my_author --name "My Author" --description "Test author"
```

### Step 2: Build a Training Dataset

Create training examples that represent your writing style:

```bash
python -m cli.main dataset build my_author
```

This opens an interactive menu where you can:

1. **Add examples from writing samples** - Paste your existing writing and get AI-suggested prompts
2. **Generate examples from prompts** - Create prompt/response pairs
3. **Import from text file** - Upload a text file to extract examples with AI prompt assistance
4. **Review current dataset** - See what examples you have so far
5. **Generate more examples from existing examples** - Use AI to create additional examples based on your existing ones (requires OpenAI API)

**Minimum recommendation**: Add 10-20 examples for basic fine-tuning.

#### AI-Assisted Prompt Creation

When adding examples from writing samples or importing from files, Simtune offers two ways to create prompts:

**Option 1: Write your own prompt (free)**
- Manually create a prompt that would generate your writing sample
- Full control over prompt wording and specificity

**Option 2: Get AI-suggested prompt (~$0.001)**  
- AI analyzes your writing sample and suggests an appropriate prompt
- Shows cost estimate before making API calls
- You can accept, edit, or reject the AI suggestion
- Automatically falls back to manual entry if AI fails

**Example workflow:**
```
You provide: "Working from home requires discipline. Set up a dedicated workspace, 
establish boundaries, and use productivity tools to stay focused."

AI suggests: "Write practical tips for maintaining productivity while working from home"

You can: Accept, edit to "Write 3-5 practical tips for remote work productivity", or write manually
```

### Step 2.5: Generate More Examples with AI (Optional)

Once you have at least 2 or 3 examples, you can use AI to generate additional examples that match your writing style:

```bash
# Select option 5 in the dataset builder menu
# This feature will:
# - Analyse your first 3 existing examples
# - Generate 1-20 new examples using OpenAI API
# - Let you review, edit, or skip each generated example
# - Add approved examples to your dataset
```

**Benefits:**

- Quickly expand your dataset from a few seed examples
- Maintains consistency with your established writing style
- Reduces manual work while ensuring quality

**Cost:** Approximately $0.002-0.01 per generated example (OpenAI API charges apply)

**Benefits of AI-Assisted Prompts:**
- Reduces cognitive load when creating training examples
- Generates more effective prompts than manual writing
- Speeds up dataset building, especially for bulk text imports
- Provides educational value - learn from AI-suggested prompts
- Optional feature - manual entry always available

### Step 3: Validate Your Dataset

Check if your dataset is ready for fine-tuning:

```bash
python -m cli.main dataset validate my_author
```

This checks:

- Dataset size (warns if < 10 examples)
- Format correctness
- Content quality
- Diversity of examples
- Safety concerns

### Step 4: Fine-tune the Model

Start the fine-tuning process:

```bash
python -m cli.main train start my_author
```

**Note**: This requires a valid OpenAI API key and will incur costs (~$8-12 for a small dataset).

Options:

- `--wait` - Wait for training to complete (20+ minutes)
- `--model gpt-4o-mini` - Specify base model (default)

### Step 5: Generate Content

Once training is complete, test your fine-tuned model:

```bash
# Single generation
python -m cli.main generate text my_author --prompt "Write about productivity"

# Interactive session
python -m cli.main generate interactive my_author
```

## Monitoring and Management

### Check Overall Status

```bash
python -m cli.main status
```

Shows all authors, dataset sizes, and training status.

### View Author Details

```bash
python -m cli.main author show my_author
```

### Check Training Progress

```bash
python -m cli.main train status my_author
```

### View Dataset Information

```bash
python -m cli.main dataset show my_author
```

## Testing Scenarios

### Scenario 1: Complete Happy Path

1. Run `init` and create author with guided setup
2. Build dataset with 5-10 examples using different input methods
3. Use AI generation (option 5) to expand to 15+ examples
4. Validate dataset (should show "READY" or "ACCEPTABLE")
5. Start training and wait for completion
6. Generate content with various prompts
7. Verify generated content matches author's style

### Scenario 2: Error Handling

1. Try commands without setting up API key (should show clear error)
2. Try to train with insufficient dataset (< 10 examples)
3. Try to generate content without a trained model
4. Test with invalid author names

### Scenario 3: AI-Assisted Prompt Creation

1. Create author and select "Add examples from writing samples" (option 1)
2. Paste a writing sample and choose "Get AI-suggested prompt" (option 2)
3. Test all three AI response options: accept, edit, reject
4. Try with different writing styles (formal, casual, technical)
5. Test API error handling by temporarily using invalid API key
6. Import text file and test AI prompts for multiple sections
7. Compare AI-suggested prompts vs manual prompts for quality

### Scenario 4: AI Generation Feature

1. Create author and add 3-5 examples manually
2. Use AI generation (option 5) to create 5-10 additional examples
3. Test review process: accept some, edit some, skip some
4. Verify generated examples maintain style consistency
5. Test with insufficient examples (< 2) to see error handling

### Scenario 5: Dataset Management

1. Create multiple authors with different styles
2. Test dataset import from text files
3. Clear and rebuild datasets
4. Export datasets to JSONL format
5. Test AI generation with different types of existing examples

## Expected Behavior

### ✅ Success Indicators

- Author creation completes without errors
- Dataset building saves examples correctly
- Validation shows "READY FOR FINE-TUNING" or "ACCEPTABLE WITH CAUTION"
- Training starts and completes successfully
- Generated content reflects the author's style

### ⚠️ Known Limitations (Stage 1 POC)

- Only supports OpenAI fine-tuning (gpt-3.5-turbo)
- No feedback/improvement loop yet
- Limited to local file storage
- Basic safety checks only
- No web UI (CLI only)

## Troubleshooting

### Common Issues

**"OpenAI API key not found"**

- Check that `.env` file exists and contains valid API key
- Ensure `.env` is in the project root directory

**"No fine-tuned model found"**

- Complete training first with `train start`
- Check training status with `train status`

**"Dataset validation failed"**

- Add more examples (minimum 10 recommended)
- Check example format in dataset review

**"Permission denied" or directory errors**

- Ensure write permissions in project directory
- Try running from project root

### Getting Help

**View command help:**

```bash
python -m cli.main --help
python -m cli.main author --help
python -m cli.main dataset --help
```

**Check logs:** Look for error messages in terminal output - they provide specific guidance.

## Cost Expectations

Fine-tuning costs with OpenAI (as of 2024):

- Small dataset (10-50 examples): ~$3-8
- Medium dataset (50-100 examples): ~$8-15
- Token costs for generation: ~$0.01-0.02 per 1000 tokens

**AI-assisted features:**
- AI-suggested prompts: ~$0.001 per prompt suggestion
- AI-generated examples: ~$0.002-0.01 per generated example
- Both features show cost estimates before making API calls

## Success Criteria

The Stage 1 POC is working correctly if:

1. ✅ You can create an author profile in under 5 minutes
2. ✅ Dataset building is intuitive and supports multiple input methods
3. ✅ Validation provides clear feedback on dataset quality
4. ✅ Fine-tuning completes without technical errors
5. ✅ Generated content noticeably reflects the input writing style
6. ✅ Error messages are helpful and actionable
7. ✅ The complete workflow takes under 30 minutes (excluding training time)

---

**Ready to test?** Start with `python -m cli.main init` and follow the prompts!
