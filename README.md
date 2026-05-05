# DocMind: Intelligent Document Management System
Project Summary
DocMind is a comprehensive document management system that combines advanced retrieval techniques with AI-powered assistance. The system uses LlamaIndex and LightRAG to provide efficient and accurate document search capabilities through an intuitive chatbot interface.
Key Components
Document Ingestion Pipeline


## Processes various document formats (PDF, TXT, DOCX)
Extracts text and metadata
Segments documents into nodes for efficient retrieval
Multiple Indexing Methods


Vector index for semantic search
Knowledge graph for relationship-based queries
BM25 for keyword search
Advanced Retrieval with LightRAG


Hybrid retrieval combining multiple strategies
Re-ranking for improved relevance
Query transformation for better search results
Intelligent Agent


Tool-based architecture for flexible capabilities
Wikipedia integration for external knowledge
Document summarization and knowledge graph navigation
User Interface


Gradio-based chat interface
Document upload functionality
Search method selection
System Performance
Based on our benchmark tests, the system demonstrates strong performance across various query types. The hybrid retrieval method generally offers the best balance of speed and accuracy, particularly for complex queries that require both semantic understanding and keyword matching.
Next Steps and Future Enhancements
Content Management Improvements


Version control for documents
Collaborative editing and annotation
User permissions and access control
Advanced AI Features


Personalized responses based on user history
Multi-document summarization
Support for multi-modal queries (text + images)
Retrieval Enhancements


Implement query routing to automatically select optimal retrieval method
Dynamic chunk size based on document type
Entity-based retrieval for specialized domains
Scalability Improvements


Implement database storage for document metadata
Add support for distributed indexing
Optimize for large document collections (1000+ documents)
User Experience


Develop mobile-friendly interface
Add visualization for knowledge graphs
Implement document recommendations
Enterprise Integration


Single Sign-On (SSO) support
Integration with popular document storage systems (SharePoint, Google Drive)
Audit logging and compliance features
Implementation Roadmap
Short-term (1-3 months)
Implement user authentication
Add document metadata management
Improve search result visualization
Create comprehensive test suite
Medium-term (3-6 months)
Develop advanced analytics dashboard
Add support for more document formats (including spreadsheets and presentations)
Implement collaborative features
Optimize for production deployment
Long-term (6-12 months)
Develop domain-specific versions (legal, medical, technical)
Add multi-language support
Implement advanced AI-powered document analysis
Create enterprise integration packages
Conclusion
DocMind represents a significant advancement in document management and retrieval technology. By combining multiple indexing strategies with intelligent agents and a user-friendly interface, the system provides an efficient solution for navigating large document collections.
The modular architecture allows for continuous improvement and customization to meet specific use cases. As language models and retrieval techniques continue to evolve, DocMind can easily incorporate these advancements to provide even better performance and user experience.





    Migrated checkpoint persistence from in-memory storage to a MongoDB-backed implementation (MongoDbStorage) within the LangGraph workflow, enabling durable state management, improved scalability, and fault-tolerant recovery across sessions.
* Replaced the previous fixed-window message trimming strategy (dropping messages beyond 10) with a rolling summarization approach, where the 10th and 11th messages are condensed into a summary and persisted as the 10th state entry, preserving conversational context while controlling state size.

Executive summary: you could say something like: "To maintain contextual awareness in long conversations, our chatbot now employs active summarization of past messages instead of basic trimming. This ensures a contiguous historical context remains, preserving vital detail and intent from previous turns, which enables better responses and prevents misunderstandings, unlike trimming where valuable information is permanently lost."


To explain this to your manager, you can present it as a **"Rolling Context Compression"** strategy. Instead of just deleting old data, you are distilling it.
Here is a brief breakdown of how the process works:
### **Implementation Strategy: Rolling Context Summarization**
The goal is to stay within the **10-node limit** of the LangGraph state while ensuring no critical information is lost.
 * **Trigger Mechanism:** When the conversation reaches the **11th message**, the system triggers a "compression" step rather than a "deletion" step.
 * **The Distillation Process:** A background LLM call takes the **oldest two messages** (the original 1st and 2nd nodes) and merges them into a single, high-density **Summary Node**.
 * **Hybrid Content Capture:** The summarization isn't just a paragraph of text. It specifically extracts and preserves:
   1. **Structured Data:** Key entities (names, dates, prices) and tool results are kept in a structured format (JSON-like).
   2. **Narrative Context:** A 1–2 line summary of the general intent or discussion topic.
 * **Re-Insertion:** This new "Summary + Entity" node is placed at the **10th position** (the back of the history). As the conversation moves forward, the "Summary Node" continues to absorb the oldest messages, acting as a "rolling snowball" of the entire conversation history.
### **Key Benefit for the Manager:**
> "This approach allows the model to 'remember' the user's initial goals even 50 messages later, without increasing the cost or latency associated with processing a massive, unorganized chat history."
> 


Incorporating the tool's JSON output into your structured memory is a smart move. It transforms the chatbot from a simple "chatter" into a **data-aware assistant**.
Since you mentioned these JSON responses contain sensitive and specific data like **Household IDs** and **Portfolio Totals**, here is how you should handle that "10th node" structured storage:
### **Strategy for Structured Tool Data Storage**
Instead of a generic text summary, your 10th node should function as a **"Conversation State Object."**
 1. **Selective Extraction (The "Signal vs. Noise" Filter):**
   Don't store the entire raw JSON if it's massive. Extract only the "Source of Truth" fields:
   * **Identifiers:** Household ID, Account Numbers.
   * **Values:** Portfolio Totals, Balances.
   * **Timestamp:** The captured_at ISO string.
 2. **The 30-Minute Expiry Logic:**
   Since financial data (like portfolio totals) can fluctuate, you should add a logic check when the LLM reads that 10th node:
   * If current_time - captured_at < 30 minutes: The LLM treats the data as **current**.
   * If current_time - captured_at > 30 minutes: The LLM is instructed to say, *"I have your portfolio data from earlier, but it may be outdated. Would you like me to refresh it?"*
### **Structure of the 10th Node (The "Rolling Memory")**
Your 10th node in the LangGraph state should look something like this:
```json
{
  "summary": "User inquired about account balances and portfolio health.",
  "structured_data": {
    "household_id": "HH_99821",
    "active_accounts": ["Acct_123", "Acct_456"],
    "total_portfolio_value": "$1,240,500",
    "data_freshness": "2026-05-04T22:45:00Z"
  },
  "entities": ["John Doe", "Retirement Fund"]
}

```
### **Why this is better than "Just a Summary":**
 * **Precision:** An LLM might hallucinate a number in a text summary (e.g., changing $1.2M to $1.1M). In a structured JSON field, the number remains exact.
 * **Context Continuity:** If the user scrolls up and asks a question 40 minutes later, the bot knows exactly which Household ID it was looking at, preventing it from having to ask "What was your account number again?"
 * **Tool Efficiency:** If the user asks a follow-up question within that 30-minute window, your system can skip a redundant API call by pulling the data directly from the structured 10th node.
**Would you like me to show you a logic snippet for how to calculate that 30-minute "freshness" check within your graph?**

Managing topic shifts is critical to prevent "contextual bleeding," where the chatbot gets confused by mixing details from an old topic (e.g., portfolio analysis) into a new one (e.g., tax planning).
To handle this within your 10-node rolling architecture, you can implement a **Topic-Aware Summary Node**.
### **1. Topic Classification at the Router Level**
Before updating the state, use a lightweight LLM call or a classifier to determine if the user's latest query belongs to the **Current Topic** or a **New Topic**.
 * **Current:** Keep appending to the existing summary.
 * **New:** Archive the current summary into a "Past Topics" block and start a fresh summary for the active discussion.
### **2. Structured Multi-Topic JSON**
Instead of a single text block, structure your 10th node to separate "Global" data from "Topic-Specific" data. This prevents the bot from losing track of the Household ID just because the user started talking about something else.
```json
{
  "global_context": {
    "household_id": "HH_99821",
    "user_name": "Pankaj",
    "last_tool_refresh": "2026-05-04T22:45:00Z"
  },
  "active_topic": {
    "label": "Estate Planning",
    "summary": "User is asking about beneficiary updates for their 401k.",
    "status": "in_progress"
  },
  "topic_history": [
    {
      "label": "Portfolio Review",
      "summary": "Reviewed $1.2M portfolio; user concerned about tech sector exposure.",
      "resolved": true
    }
  ]
}

```
### **3. The "Decay" Strategy for Summarization**
When a topic shift is detected, apply a **relevance decay** to the old summary:
 * **Active Topic:** Detailed (3–4 sentences + specific entities).
 * **Previous Topic:** Distilled (1 sentence summary).
 * **Older Topics:** Keyword only (e.g., "Discussed: 2025 Tax Filing").
This ensures that the 10th node doesn't grow too large while still providing "hooks" for the LLM to realize it has talked about these things before.
### **4. Addressing the Shift in the Prompt**
When the summary is injected into the 11th message context, the System Prompt should be instructed on how to use it:
> *"The provided summary contains a 'Global Context' (stable facts) and 'Topic History'. If the user asks a question about a past topic, reference the 'Topic History' to provide continuity. If they are on a new topic, focus on the 'Active Topic' but remain aware of global constraints."*
> 
### **Executive Benefit for your Manager:**
> "By implementing **Topic-Segmented Summarization**, the system maintains a 'Mental Map' of the conversation. This prevents the model from hallucinating or mixing data between unrelated financial tasks, ensuring high precision even when the user switches rapidly between disparate topics like market trends and personal tax records."
> 
Does this segmented approach fit within the logic you've already built for your LangGraph nodes?



