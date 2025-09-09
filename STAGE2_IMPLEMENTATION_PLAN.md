# Stage 2: Browser-Based UI Implementation Plan

## Architecture Overview

**Deployment Choice: Netlify**
- Free tier with generous limits
- Custom headers for CORS handling
- Netlify Functions available for future server-side needs
- Excellent developer experience and CI/CD integration

**Frontend Framework: Svelte + SvelteKit**
- Smaller bundle sizes = faster loading for users
- Modern reactive programming model
- Easy learning curve for contributors
- Built-in state management
- Server-side rendering capabilities for future scaling

**Storage Strategy: Client-Side with Export/Import**
- IndexedDB for structured data storage (author profiles, datasets, fine-tune jobs)
- localStorage for API keys and user preferences
- Export/Import functionality for data portability and backup
- Migration tools from Stage 1 CLI data format

## Implementation Structure

```
web/
├── src/
│   ├── lib/
│   │   ├── components/         # Reusable UI components
│   │   │   ├── AuthorCard.svelte
│   │   │   ├── DatasetBuilder.svelte
│   │   │   ├── FineTuneStatus.svelte
│   │   │   └── GenerationPanel.svelte
│   │   ├── stores/             # Svelte stores for state management
│   │   │   ├── authors.js
│   │   │   ├── datasets.js
│   │   │   ├── models.js
│   │   │   └── settings.js
│   │   ├── services/           # Business logic layer
│   │   │   ├── AuthorService.js
│   │   │   ├── DatasetService.js
│   │   │   ├── OpenAIService.js
│   │   │   └── StorageService.js
│   │   └── utils/              # Utility functions
│   │       ├── validation.js
│   │       ├── crypto.js
│   │       └── export.js
│   ├── routes/                 # SvelteKit pages
│   │   ├── +layout.svelte      # Main app layout
│   │   ├── +page.svelte        # Dashboard/home
│   │   ├── authors/
│   │   │   ├── +page.svelte    # Author management
│   │   │   └── [id]/
│   │   │       ├── +page.svelte        # Author detail
│   │   │       ├── dataset/+page.svelte # Dataset builder
│   │   │       ├── train/+page.svelte   # Fine-tuning
│   │   │       └── generate/+page.svelte # Content generation
│   │   └── settings/+page.svelte       # API keys & preferences
│   └── app.html                # HTML template
├── static/                     # Static assets
├── package.json
├── svelte.config.js
├── vite.config.js
└── README.md
```

## Core Functionality Implementation

### 1. Data Migration & Compatibility
- **JSON Schema Validation**: Port Pydantic models to JavaScript validation
- **Import CLI Data**: File upload to import existing Stage 1 author profiles and datasets
- **Export Functionality**: Download data as ZIP with all author files (JSON, JSONL, MD)
- **Data Versioning**: Handle schema evolution between CLI and web versions

### 2. API Key Management
- **Storage Options**:
  - localStorage (default, simplest)
  - Optional client-side encryption with user-provided passphrase
  - Session-only storage for temporary usage
- **Security Measures**:
  - Clear warnings about browser storage limitations
  - Easy key deletion/rotation
  - API key validation before storage
  - No transmission to any third-party services (keys go directly to OpenAI)

### 3. Browser Storage Strategy
- **IndexedDB Implementation**:
  - `authors` table: AuthorProfile objects
  - `datasets` table: Dataset objects with training examples
  - `models` table: ModelMetadata with fine-tune job history
  - `content` table: Generated content with metadata
- **Storage Limits**: Monitor usage, warn users near browser limits
- **Data Integrity**: Validation on read/write, corruption recovery

### 4. OpenAI Integration
- **Direct API Calls**: Fetch from browser to OpenAI (CORS-enabled)
- **File Upload Handling**: Convert datasets to proper JSONL format
- **Progress Tracking**: Real-time status updates for fine-tuning jobs
- **Error Handling**: User-friendly error messages with actionable guidance

### 5. UI/UX Design
- **Responsive Design**: Mobile-first approach using CSS Grid/Flexbox
- **Progressive Enhancement**: Works without JavaScript for basic functionality
- **Accessibility**: WCAG 2.1 AA compliance
- **Theme Support**: Light/dark mode toggle
- **Loading States**: Skeleton screens and progress indicators

## Page-by-Page Implementation

### Dashboard (`/`)
- Author overview cards with status indicators
- Recent activity feed (generations, training jobs)
- Quick actions (new author, generate content)
- Storage usage visualization

### Author Management (`/authors`)
- Author grid with search/filter capabilities
- Create new author wizard
- Bulk operations (export, delete)
- Import authors from CLI data

### Author Detail (`/authors/[id]`)
- Author profile editor with style guide configuration
- Navigation tabs: Dataset, Training, Generation, Content
- Real-time stats (dataset size, model status, generation count)

### Dataset Builder (`/authors/[id]/dataset`)
- Interactive example creation with preview
- Bulk import from various formats (TXT, MD, JSON)
- Dataset validation with quality metrics
- Export dataset in OpenAI format

### Fine-tuning (`/authors/[id]/train`)
- Training job configuration with hyperparameter controls
- Real-time progress monitoring with logs
- Job history with success/failure analysis
- Model comparison tools

### Content Generation (`/authors/[id]/generate`)
- Single generation with customizable parameters
- Chat-like interface for interactive generation
- Content history with search and filtering
- Export generated content (individual or bulk)

### Settings (`/settings`)
- API key management with validation
- Export/import global settings
- Data management (clear all data, export backup)
- App preferences (theme, notifications)

## Technical Implementation Details

### 1. State Management
- **Svelte Stores**: Reactive state containers for each data type
- **Persistent Stores**: Auto-sync with IndexedDB
- **Derived Stores**: Computed values (e.g., training progress percentages)

### 2. API Service Layer
```javascript
class OpenAIService {
  constructor(apiKey) { this.apiKey = apiKey; }
  
  async uploadFile(dataset) { /* Upload JSONL to OpenAI */ }
  async createFineTune(params) { /* Start fine-tuning job */ }
  async checkJobStatus(jobId) { /* Poll job status */ }
  async generateText(model, prompt, options) { /* Generate content */ }
}
```

### 3. Data Validation
- **Zod Schema**: Runtime type checking (JavaScript equivalent of Pydantic)
- **Form Validation**: Real-time input validation with helpful error messages
- **Data Sanitization**: Prevent XSS and ensure data integrity

### 4. Progressive Web App Features
- **Service Worker**: Offline functionality for dataset building
- **App Manifest**: Installable web app experience
- **Push Notifications**: Optional training completion notifications

## Deployment Pipeline

### 1. Development Setup
```bash
npm create svelte@latest simtune-web
cd simtune-web
npm install
npm run dev
```

### 2. Netlify Deployment
- **Build Command**: `npm run build`
- **Publish Directory**: `build/`
- **Environment Variables**: None required (client-side app)
- **Custom Headers**: Enable CORS for OpenAI API calls

### 3. CI/CD Pipeline
- **GitHub Actions**: Auto-deploy on push to main
- **Testing**: Unit tests, integration tests, E2E tests with Playwright
- **Lighthouse Checks**: Performance, accessibility, SEO scoring
- **Bundle Analysis**: Monitor bundle size and performance

## Migration Strategy from Stage 1

### 1. Data Migration Tools
- **CLI Export Command**: Add web-compatible export to existing CLI
- **Web Import Interface**: Drag-and-drop ZIP file import
- **Data Validation**: Ensure compatibility between versions

### 2. Feature Parity Checklist
- ✅ Author profile creation and management
- ✅ Style guide configuration
- ✅ Dataset building with various input methods
- ✅ Training data validation and export
- ✅ Fine-tuning job creation and monitoring
- ✅ Content generation (single and interactive)
- ✅ Content persistence and management
- ✅ Data export for portability

### 3. Documentation Updates
- **User Guide**: Web-specific usage instructions
- **Migration Guide**: Step-by-step CLI to web transition
- **API Documentation**: For potential integrations

## Security Considerations

### 1. API Key Protection
- **Local Storage Only**: Keys never transmitted to Simtune servers
- **Encryption Option**: Client-side encryption with user passphrase
- **Clear Warnings**: Educate users about browser storage risks
- **Key Rotation**: Easy key update interface

### 2. Data Privacy
- **Local-First**: All data remains in user's browser
- **Export Control**: Users own and control their data
- **No Analytics**: No user tracking or data collection
- **Open Source**: Full transparency of data handling

## Future Extensibility

### 1. Stage 3+ Preparation
- **Plugin Architecture**: Extensible for multiple LLM providers
- **API Abstraction**: Easy to add Gemini, Anthropic, etc.
- **Modular Design**: Components can be reused in different contexts

### 2. Advanced Features
- **Collaborative Editing**: Share authors between users (Stage 4+)
- **Version Control**: Track changes to authors and datasets
- **Analytics Dashboard**: Usage statistics and insights
- **Integration APIs**: Connect with writing tools and platforms

## Success Metrics

### 1. Technical Metrics
- **Load Time**: < 3 seconds on 3G connection
- **Bundle Size**: < 500KB initial load
- **Lighthouse Score**: > 90 across all categories
- **Browser Support**: Modern browsers (ES2020+)

### 2. User Experience Metrics
- **Feature Parity**: 100% of CLI functionality available
- **Data Migration**: Seamless import from Stage 1
- **Accessibility**: WCAG 2.1 AA compliance
- **Mobile Experience**: Fully responsive design

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- Set up SvelteKit project structure
- Implement core data models and validation
- Create IndexedDB storage layer
- Build basic routing and layout

### Phase 2: Core Features (Weeks 3-6)
- Author management (CRUD operations)
- Dataset builder with import/export
- OpenAI service integration
- Basic UI components

### Phase 3: Advanced Features (Weeks 7-10)
- Fine-tuning workflow with progress tracking
- Content generation interfaces
- Data migration tools from CLI
- Responsive design and accessibility

### Phase 4: Polish & Deploy (Weeks 11-12)
- Performance optimization
- Comprehensive testing
- Documentation and user guides
- Netlify deployment and CI/CD

This implementation plan provides a comprehensive browser-based UI that maintains all Stage 1 functionality while being deployable on static hosting platforms with secure API key management.