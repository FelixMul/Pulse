# Plan: Synthetic Dataset Generation (V2 - Enhanced Realism)

## 1. Objective

To create a synthetic dataset of 1,250 emails that realistically simulate the inbox of a UK Member of Parliament. This dataset will be used to test and fine-tune our project's NLP models, including email cleaning, topic classification, and sentiment analysis. The final output will be a structured data file (e.g., CSV).

## 2. Core Ingredients

We will generate each email based on a combination of four key ingredients: Persona, Sentiment, Length, and Theme(s).

### a) Personas

We will create a comprehensive list of around **15-20 distinct personas** to ensure a wide variety of writing styles. These will be relevant to a UK constituency. Examples include:

*   "An angry resident from a village concerned about the lack of local services."
*   "A concerned but polite parent from a suburban area writing about school issues."
*   "A polite pensioner worried about the cost of living and their energy bills."
*   "A frustrated local shop owner complaining about business rates."
*   "A university student politely requesting information for a research project."
*   "An NHS nurse writing passionately about working conditions."
*   "A person using informal language, some slang, and occasional typos."
*   "A paranoid individual writing a hard-to-follow, incoherent email."

### b) Sentiments

A list of 6 specific sentiments to label each email:

*   "Very Negative"
*   "Negative"
*   "Neutral"
*   "Positive"
*   "Very Positive"
*   "Mixed"

### c) Email Length

To ensure variety, we will specify the desired length of the email body with the following distribution over the 1,250 emails:

*   **Short** (1-3 sentences): **10%** (125 emails)
*   **Medium** (2-4 paragraphs): **80%** (1000 emails)
*   **Long** (5+ paragraphs): **10%** (125 emails)

### d) Themes & Combination Strategy

We will define a list of around **15 core, high-level themes** relevant to UK politics. To create realistic emails, we will not combine them randomly. Instead, we will create a master list of **pre-defined, logical topic combinations**.

*   **The Recipe:** Our script will sample from this master list of combinations to assign topics to each of the 1,250 emails according to the following distribution:
    *   **70% (875 emails):** Will be assigned a **single** theme.
    *   **25% (313 emails):** Will be assigned a pre-defined **pair** of themes.
    *   **5% (62 emails):** Will be assigned a pre-defined **triplet** of themes.
*   **Special Categories:** A small portion of the recipes (approx. 3-5%) will be explicitly dedicated to special 'catch-all' categories like `"Non-Actionable / Incoherent"` to ensure our model learns to identify and discard "weird mail."

## 3. The Three-Phase Generation Process

Our workflow is broken into three distinct phases to ensure quality and realism.

### Phase 1: Recipe Creation

First, we will programmatically create a "recipe" dataframe containing 1,250 rows. This dataframe will pre-determine the parameters for every email we generate.

*   **Action:** Write a Python script
*   **Output:** A dataframe with the columns: `request_id` (this will need to be included by the LLM in their response to be able to match it with the input parameters), `persona`, `sentiment`, `length`, and `topics` (this column will contain a list of one, two, or three pre-combined topics).

### Phase 2: Clean Content Generation via Gemini API

With our recipes ready, we will generate the core email text.

1.  **Single Test Request:** We will construct a single, detailed prompt using the first row of our recipe dataframe. We will send this to the **Google Gemini Flash model** to ensure the API connection works and the output format (a JSON object with "email_header" and "email_body") is correct.

2.  **Full Batch Generation:** Once validated, we will iterate through our entire recipe dataframe and submit all 1,250 generation requests to the Gemini API using **Batch Mode** for cost efficiency.

*   **Output:** A list of 1,250 "clean" and grammatically perfect email headers and bodies.

### Phase 3: Realism Injection & Final Assembly

This is a critical final step. We will programmatically "degrade" the perfect LLM-generated text to mimic real-world emails.

2.  **Final Assembly:** We will merge the now-realistic `email_header` and `email_body` back into our recipe dataframe.

*   **Final Output:** A dataframe (saved as a CSV file) with the columns: `persona`, `sentiment`, `length`, `topics`, `email_header`, `email_body`. This is our final, high-quality synthetic dataset.

## 4. Our First Step

Our immediate next step is to write the Python code for **Phase 1: Recipe Creation**. We will need to define the full lists of personas and themes, and then build the initial 1,250-row dataframe according to the specified distributions. Do not start coding until we confirm this plan.