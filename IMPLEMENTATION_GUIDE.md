# Parliament Pulse POC - Implementation Steps

## 🎯 POC Goals
Create a proof-of-concept demonstrating:
1. **Local LLM analysis** for topic extraction and sentiment analysis using gpt-oss 20B
2. **Enhanced test UI** with time-series visualizations
3. **Time-series analysis** showing topic and sentiment trends over time
4. **Simple deployment** with streamlined tooling

## 🔧 Technical Stack

### Backend
- **Ollama** - Local LLM inference server
- **LLM** - gpt-oss 20B (final choice)
- **FastAPI** - Simplified API server
- **Local JSON storage** - No complex database needed
- **UV** - Fast Python package manager

### Frontend
- **Enhanced test-ui** - Improved existing HTML/JS interface
- **Chart.js** - Time-series line plots
- **Modern CSS** - Professional styling improvements

### Development
- **Single terminal deployment** - Run both backend and frontend together
- **UV for dependency management** - Faster than pip
- **Simplified project structure** - Remove unused components

---

## 📋 Implementation Steps

### ✅ Step 1: Simplify Development Environment ✅ COMPLETED
**Goal:** Single terminal deployment with UV package management

#### 1.1: Install UV and Setup Dependencies
```bash
# Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Convert from pip to UV
uv init --no-readme
uv add fastapi uvicorn httpx

# Single command to run both backend and frontend
uv run python start_server.py
```

#### 1.2: Streamlined Project Structure
- Remove: `frontend/`, `ModelFinetuning/`, OAuth files
- Keep: `backend/app/`, `test-ui/`, core functionality
- Simplify: Requirements, configuration, startup

#### 1.3: Unified Startup Script
Create `start_server.py` that runs:
- FastAPI backend on port 8001
- Static file server for test-ui on port 8080
- Auto-opens browser to test interface

**Status:** ✅ COMPLETED ✅

#### 1.4: What Was Accomplished
- ✅ Removed unused components (ModelFinetuning/, OAuth files)
- ✅ Created pyproject.toml for UV package management
- ✅ Built unified start_server.py for single-command deployment
- ✅ Simplified backend/app/config.py (removed OAuth settings)
- ✅ Created backend/app/simple_storage.py (JSON-based local storage)
- ✅ Streamlined backend/app/main.py (removed complex dependencies)
- ✅ Updated test-ui/index.html to work with simplified API
- ✅ Updated README with new startup instructions
- ✅ Fixed CSV data serving for test interface

#### 1.5: New Startup Process
```bash
# Install dependencies
uv sync

# Start everything in one command
uv run python start_server.py

# Opens browser to: http://localhost:8080
```

#### 1.6: Verified Working
- ✅ Single command startup
- ✅ Test interface loads emails from CSV
- ✅ Mock analysis endpoint working
- ✅ Local storage saving results
- ✅ Ready for LLM integration

---

### ✅ Step 2: Local LLM Integration ✅ COMPLETED
**Goal:** gpt-oss 20B for reliable topic extraction and sentiment analysis

#### 2.1: Install Ollama and gpt-oss 20B
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull gpt-oss 20B model  
ollama pull gpt-oss:20b

# Test the model
ollama run gpt-oss:20b "Analyze this email sentiment: I'm frustrated with healthcare delays"
```

#### 2.2: What Was Accomplished
- ✅ Created LLMProcessor class with gpt-oss:20b integration
- ✅ Implemented structured prompting for reliable JSON output
- ✅ Added comprehensive error handling and fallback mechanisms
- ✅ Integrated LLM processor into main analyze endpoint
- ✅ Added LLM health checks and connection testing
- ✅ Validated topic extraction and sentiment analysis accuracy

#### 2.3: LLM Features Implemented
- **Topic Classification**: 29 comprehensive categories (healthcare, education, transportation, etc.)
- **Sentiment Analysis**: 5-level scale (very_negative to very_positive)
- **Confidence Scoring**: 0.1 to 1.0 range with validation
- **Summary Generation**: Concise 1-sentence email summaries
- **Error Recovery**: Graceful fallbacks when LLM fails
- **JSON Validation**: Robust parsing and field validation

#### 2.4: Critical Bug Fixes Completed ✅
- **CSV Parsing Fix**: Fixed multi-line email parsing (was only reading first line like "Dear Mr. Johnson,")
- **Email Truncation Fix**: Increased character limit from 1500 to 4000 chars (74% of emails were being cut off)
- **Robust JSON Extraction**: Added regex-based parsing to handle LLM "thinking mode" output
- **Comprehensive Debugging**: Added terminal debugging to trace full analysis pipeline
- **Email Caching**: Implemented email_id-based caching to avoid redundant LLM calls

#### 2.5: Performance Metrics ✅
- **Average Email Length**: 1948 characters (now processed in full)
- **Email Truncation**: 74.4% of emails were over 1500 chars (now handled properly)
- **LLM Success Rate**: Significantly improved with full email context and robust JSON parsing
- **Frontend Integration**: Complete email analysis with topic, sentiment, confidence, and summary
- **Caching System**: Working properly with visual "✓ Analyzed" indicators

#### 2.4: Tested Working Examples
```bash
# Healthcare complaint (very_negative)
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Frustrated with hospital waiting times..."}'

# Transportation praise (positive)  
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Thank you for improving bus routes..."}'
```

**Status:** ✅ COMPLETED ✅

### Step 2: MCP Integration

#### 2.1: MCP Server Setup
```python
# backend/app/mcp_server.py
class MCPEnhancer:
    """Enhances email analysis with external context"""
    
    async def get_constituency_data(self, postcode: str) -> dict:
        # Mock API call to constituency demographics
        pass
        
    async def get_policy_context(self, topic: str) -> dict:
        # Mock API call to current policy positions
        pass
        
    async def get_news_sentiment(self, topic: str, date: str) -> dict:
        # Mock API call to news sentiment for topic
        pass
        
    async def enhance_analysis(self, email_data: dict) -> dict:
        # Combine LLM analysis with external context
        pass
```

#### 2.2: API Endpoints
```python
# Add to backend/app/main.py
@app.post("/api/mcp/analyze")
async def enhanced_analysis(email_data: dict):
    # Run LLM analysis + MCP enhancement
    pass

@app.get("/api/mcp/status") 
async def mcp_status():
    # Health check for MCP services
    pass
```

### Step 3: Synthetic Time-Series Data

#### 3.1: Realistic Data Generation
```python
# backend/app/synthetic_data.py
def generate_time_series_data():
    """Generate 6 months of realistic email data"""
    
    # Patterns to implement:
    # - Healthcare peaks in winter (Nov-Feb)
    # - Education surges in Aug-Sep  
    # - Tax negativity around budget periods
    # - Environment spikes with weather events
    # - Economy sentiment varies with news cycles
    
    # Return format:
    # [
    #   {
    #     "date": "2024-01-15",
    #     "topic": "healthcare", 
    #     "sentiment": -0.3,
    #     "volume": 45,
    #     "trend": "up"
    #   }
    # ]
```

#### 3.2: Time-Series API Endpoints
```python
# Add to backend/app/main.py
@app.get("/api/dashboard/time-series")
async def get_time_series(
    start_date: str,
    end_date: str, 
    granularity: str = "day"
):
    # Return time-series data for charts
    pass
```

### Step 4: Premium Frontend Upgrade

#### 4.1: Install Premium Dependencies
```bash
cd frontend
npm install framer-motion recharts @headlessui/react lucide-react
npm install @types/react @types/react-dom
```

#### 4.2: Component Architecture
```typescript
// frontend/src/components/TimeSeriesChart.tsx
interface TimeSeriesProps {
  data: TimeSeriesPoint[];
  selectedTopics: string[];
  dateRange: DateRange;
}

export const TimeSeriesChart: React.FC<TimeSeriesProps> = ({ data, selectedTopics, dateRange }) => {
  // Multi-line chart showing topic trends over time
  // Interactive brushing and zooming
  // Smooth animations with Recharts
};

// frontend/src/components/SentimentHeatmap.tsx
export const SentimentHeatmap: React.FC = () => {
  // Calendar heatmap showing sentiment patterns
  // Color-coded by sentiment intensity
  // Interactive hover states
};

// frontend/src/components/AnimatedKPICards.tsx  
export const AnimatedKPICards: React.FC = () => {
  // Animated metric cards with trend indicators
  // Count-up animations for numbers
  // Smooth color transitions for sentiment
};
```

#### 4.3: Advanced Animations
```typescript
// frontend/src/components/MotionLayout.tsx
import { motion, AnimatePresence } from 'framer-motion';

export const MotionLayout: React.FC = ({ children }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
};
```

#### 4.4: Theme System
```typescript
// frontend/src/hooks/useTheme.ts
export const useTheme = () => {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  
  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };
  
  return { theme, toggleTheme };
};
```

### Step 5: Time-Series Visualizations

#### 5.1: Multi-Line Trend Chart
```typescript
// Show multiple topics on same timeline
// Interactive legend to toggle topics
// Zoom and pan functionality
// Gradient fills under lines
// Custom tooltips with rich content
```

#### 5.2: Correlation Analysis
```typescript  
// Scatter plot showing topic vs sentiment correlation
// Bubble size represents email volume
// Color coding by time period
// Interactive selection and filtering
```

#### 5.3: Seasonal Pattern Display
```typescript
// Identify and highlight seasonal trends
// Show year-over-year comparisons
// Pattern recognition annotations
// Predictive trend lines
```

### Step 6: Integration and Testing

#### 6.1: End-to-End Flow
```python
# Test complete pipeline:
# Email text → LLM analysis → MCP enhancement → Database → API → Frontend charts
```

#### 6.2: Performance Optimization
```python
# LLM response caching
# Batch processing for multiple emails  
# Lazy loading for large datasets
# Chart rendering optimization
```

#### 6.3: Error Handling
```python
# LLM timeout handling
# MCP service fallbacks
# Graceful degradation for missing data
# User-friendly error messages
```

---

## 🎯 Success Criteria

✅ **LLM Analysis**: Process emails in <5 seconds with structured output  
✅ **Premium UI**: Smooth 60fps animations and professional design  
✅ **Time-Series**: Interactive charts showing trends over 6-month period  
✅ **MCP Integration**: External context enhancement working  
✅ **Synthetic Data**: Realistic patterns demonstrating system capabilities  

## 📝 Next Actions

1. **Choose LLM model** (Llama 3.2 3B vs Gemma 2B vs Mistral 7B)
2. **Set up Ollama** and test model performance
3. **Implement LLM processor** with structured prompting
4. **Create synthetic dataset** with realistic time-series patterns
5. **Build premium UI components** with Framer Motion
6. **Add MCP integration** for enhanced analysis
7. **Connect everything** and test end-to-end flow 