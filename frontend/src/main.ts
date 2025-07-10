/**
 * Parliament Pulse - Frontend Application
 * Privacy-first email analysis dashboard for parliamentarians
 */

import './style.css'
import { Chart, registerables } from 'chart.js'
import 'chartjs-adapter-date-fns'

// Register Chart.js components
Chart.register(...registerables)

// Type definitions
interface DashboardData {
  summary: {
    total_emails: number
    avg_sentiment: number
    unique_topics: number
  }
  email_trends: Array<{
    date: string
    count: number
  }>
  top_topics: Array<{
    topic: string
    count: number
  }>
  sentiment_by_topic: Array<{
    topic: string
    positive: number
    neutral: number
    negative: number
  }>
}

interface AppState {
  isLoading: boolean
  error: string | null
  startDate: string
  endDate: string
  granularity: 'day' | 'week' | 'month'
  data: DashboardData | null
  charts: {
    emailTrend?: Chart
    topics?: Chart
    sentimentTopic?: Chart
  }
}

// Global application state
const appState: AppState = {
  isLoading: true,
  error: null,
  startDate: '',
  endDate: '',
  granularity: 'week',
  data: null,
  charts: {}
}

// DOM element references
const elements = {
  loadingScreen: document.getElementById('loading-screen')!,
  app: document.getElementById('app')!,
  errorState: document.getElementById('error-state')!,
  connectionStatus: document.getElementById('connection-status')!,
  startDate: document.getElementById('start-date') as HTMLInputElement,
  endDate: document.getElementById('end-date') as HTMLInputElement,
  granularity: document.getElementById('granularity') as HTMLSelectElement,
  refreshButton: document.getElementById('refresh-data')!,
  retryButton: document.getElementById('retry-connection')!,
  errorMessage: document.getElementById('error-message')!,
  totalEmails: document.getElementById('total-emails')!,
  avgSentiment: document.getElementById('avg-sentiment')!,
  uniqueTopics: document.getElementById('unique-topics')!,
  emailTrendChart: document.getElementById('email-trend-chart') as HTMLCanvasElement,
  topicsChart: document.getElementById('topics-chart') as HTMLCanvasElement,
  sentimentTopicChart: document.getElementById('sentiment-topic-chart') as HTMLCanvasElement
}

// API client
class ApiClient {
  private baseUrl: string = '/api'

  async healthCheck(): Promise<{ status: string; oauth_configured: boolean }> {
    const response = await fetch(`${this.baseUrl}/health`)
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`)
    }
    return response.json()
  }

  async getDashboardData(startDate: string, endDate: string, granularity: string): Promise<DashboardData> {
    const params = new URLSearchParams({
      start_date: startDate,
      end_date: endDate,
      granularity
    })
    
    const response = await fetch(`${this.baseUrl}/dashboard/data?${params}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch dashboard data: ${response.statusText}`)
    }
    return response.json()
  }
}

const apiClient = new ApiClient()

// Utility functions
function formatDate(date: Date): string {
  return date.toISOString().split('T')[0]
}

function setDefaultDates(): void {
  const endDate = new Date()
  const startDate = new Date()
  startDate.setMonth(startDate.getMonth() - 6) // Default to 6 months ago
  
  appState.endDate = formatDate(endDate)
  appState.startDate = formatDate(startDate)
  
  elements.endDate.value = appState.endDate
  elements.startDate.value = appState.startDate
}

function updateConnectionStatus(connected: boolean, message: string = ''): void {
  const statusDot = elements.connectionStatus.querySelector('.w-2') as HTMLElement
  const statusText = elements.connectionStatus.querySelector('span') as HTMLElement
  
  if (connected) {
    statusDot.className = 'w-2 h-2 bg-green-500 rounded-full'
    statusText.textContent = 'Connected'
  } else {
    statusDot.className = 'w-2 h-2 bg-red-500 rounded-full'
    statusText.textContent = message || 'Disconnected'
  }
}

function showError(message: string): void {
  appState.error = message
  elements.errorMessage.textContent = message
  elements.errorState.classList.remove('hidden')
  elements.app.classList.add('hidden')
  elements.loadingScreen.classList.add('hidden')
  updateConnectionStatus(false, 'Error')
}

function hideError(): void {
  appState.error = null
  elements.errorState.classList.add('hidden')
}

function showLoading(): void {
  appState.isLoading = true
  elements.loadingScreen.classList.remove('hidden')
  updateConnectionStatus(false, 'Loading...')
}

function hideLoading(): void {
  appState.isLoading = false
  elements.loadingScreen.classList.add('hidden')
  elements.app.classList.remove('hidden')
  updateConnectionStatus(true)
}

// Chart management
function createEmailTrendChart(data: DashboardData['email_trends']): void {
  const ctx = elements.emailTrendChart.getContext('2d')!
  
  // Destroy existing chart if it exists
  if (appState.charts.emailTrend) {
    appState.charts.emailTrend.destroy()
  }
  
  appState.charts.emailTrend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map(d => d.date),
      datasets: [{
        label: 'Email Count',
        data: data.map(d => d.count),
        borderColor: '#0ea5e9',
        backgroundColor: 'rgba(14, 165, 233, 0.1)',
        tension: 0.4,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            precision: 0
          }
        }
      }
    }
  })
}

function createTopicsChart(data: DashboardData['top_topics']): void {
  const ctx = elements.topicsChart.getContext('2d')!
  
  if (appState.charts.topics) {
    appState.charts.topics.destroy()
  }
  
  appState.charts.topics = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.topic),
      datasets: [{
        label: 'Email Count',
        data: data.map(d => d.count),
        backgroundColor: '#eab308',
        borderColor: '#ca8a04',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          ticks: {
            precision: 0
          }
        }
      }
    }
  })
}

function createSentimentTopicChart(data: DashboardData['sentiment_by_topic']): void {
  const ctx = elements.sentimentTopicChart.getContext('2d')!
  
  if (appState.charts.sentimentTopic) {
    appState.charts.sentimentTopic.destroy()
  }
  
  appState.charts.sentimentTopic = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.topic),
      datasets: [
        {
          label: 'Positive',
          data: data.map(d => d.positive),
          backgroundColor: '#22c55e'
        },
        {
          label: 'Neutral',
          data: data.map(d => d.neutral),
          backgroundColor: '#6b7280'
        },
        {
          label: 'Negative',
          data: data.map(d => d.negative),
          backgroundColor: '#ef4444'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          stacked: true
        },
        y: {
          stacked: true,
          beginAtZero: true,
          ticks: {
            precision: 0
          }
        }
      }
    }
  })
}

function updateKPIs(data: DashboardData['summary']): void {
  elements.totalEmails.textContent = data.total_emails.toLocaleString()
  
  // Format sentiment score
  const sentimentScore = data.avg_sentiment
  let sentimentText = 'Neutral'
  let sentimentClass = 'text-gray-900'
  
  if (sentimentScore > 0.1) {
    sentimentText = 'Positive'
    sentimentClass = 'text-green-600'
  } else if (sentimentScore < -0.1) {
    sentimentText = 'Negative'
    sentimentClass = 'text-red-600'
  }
  
  elements.avgSentiment.textContent = sentimentText
  elements.avgSentiment.className = `text-2xl font-bold ${sentimentClass}`
  
  elements.uniqueTopics.textContent = data.unique_topics.toString()
}

// Data loading and updating
async function loadDashboardData(): Promise<void> {
  try {
    showLoading()
    hideError()
    
    const data = await apiClient.getDashboardData(
      appState.startDate,
      appState.endDate,
      appState.granularity
    )
    
    appState.data = data
    
    // Update KPIs
    updateKPIs(data.summary)
    
    // Update charts
    createEmailTrendChart(data.email_trends)
    createTopicsChart(data.top_topics)
    createSentimentTopicChart(data.sentiment_by_topic)
    
    hideLoading()
  } catch (error) {
    console.error('Failed to load dashboard data:', error)
    showError(`Failed to load data: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

// Event listeners
function setupEventListeners(): void {
  // Date and granularity change handlers
  elements.startDate.addEventListener('change', (e) => {
    appState.startDate = (e.target as HTMLInputElement).value
  })
  
  elements.endDate.addEventListener('change', (e) => {
    appState.endDate = (e.target as HTMLInputElement).value
  })
  
  elements.granularity.addEventListener('change', (e) => {
    appState.granularity = (e.target as HTMLSelectElement).value as 'day' | 'week' | 'month'
  })
  
  // Refresh button
  elements.refreshButton.addEventListener('click', loadDashboardData)
  
  // Retry button
  elements.retryButton.addEventListener('click', async () => {
    hideError()
    await initialize()
  })
}

// Application initialization
async function initialize(): Promise<void> {
  try {
    showLoading()
    
    // Check backend connectivity
    await apiClient.healthCheck()
    
    // Set default date range
    setDefaultDates()
    
    // Setup event listeners
    setupEventListeners()
    
    // Load initial data
    await loadDashboardData()
    
  } catch (error) {
    console.error('Failed to initialize application:', error)
    showError(`Failed to connect to backend: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

// Start the application
document.addEventListener('DOMContentLoaded', initialize)

// Handle browser navigation
window.addEventListener('beforeunload', () => {
  // Cleanup charts to prevent memory leaks
  Object.values(appState.charts).forEach(chart => chart?.destroy())
}) 