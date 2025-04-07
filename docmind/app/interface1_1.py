# Updated interface.py with model selection
import gradio as gr
import os
from app.agents import DocMindAgent
from app.indexing import DocumentIndexer
from app.retrieval import LightRAGRetriever
from app.ingestion import DocumentProcessor

def create_interface(agent):
    """Create a Gradio interface for interacting with the agent."""

    # Import model configs
    from app.ai_models import AVAILABLE_LLMS, AVAILABLE_EMBEDDINGS

    css = """
    .chatbot {
        height: 520px;
        overflow-y: auto;
    }
    .message {
        padding: 10px;
        margin: 5px;
        border-radius: 10px;
    }
    .user {
        background-color: #e6f7ff;
        border: 1px solid #91d5ff;
        text-align: right;
    }
    .bot {
        background-color: #f6ffed;
        border: 1px solid #b7eb8f;
        text-align: left;
    }
    """

    with gr.Blocks(css=css) as interface:
        gr.Markdown("# DocMind: Intelligent Document Search")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot([], elem_classes="chatbot")
                msg = gr.Textbox(
                    placeholder="Ask me anything about your documents...",
                    label="Message",
                    lines=2
                )

                def respond(message, history):
                    # Add user message to history
                    history.append((message, None))

                    # Get response from agent
                    response = agent.chat(message)

                    # Update history with response
                    history[-1] = (message, response)
                    return "", history

                msg.submit(respond, [msg, chatbot], [msg, chatbot])

                clear = gr.Button("Clear Chat")
                clear.click(lambda: [], None, chatbot)

            with gr.Column(scale=1):
                with gr.Accordion("Model Settings", open=False):
                    llm_dropdown = gr.Dropdown(
                        choices=list(AVAILABLE_LLMS.keys()),
                        value="claude-3-7-sonnet",
                        label="LLM Model"
                    )

                    embedding_dropdown = gr.Dropdown(
                        choices=list(AVAILABLE_EMBEDDINGS.keys()),
                        value="openai-small",
                        label="Embedding Model"
                    )

                    apply_model_btn = gr.Button("Apply Model Changes")
                    model_status = gr.Markdown("Current models: claude-3-7-sonnet & openai-small")

                    def update_models(llm_choice, embedding_choice):
                        from app.model_factory import get_llm_client, get_embedding_client
                        from app.ai_models import AVAILABLE_LLMS, AVAILABLE_EMBEDDINGS

                        # Get model info
                        llm_info = AVAILABLE_LLMS[llm_choice]
                        embed_info = AVAILABLE_EMBEDDINGS[embedding_choice]

                        try:
                            # Update agent's models
                            agent.indexer.llm = get_llm_client(
                                llm_info["provider"],
                                llm_info["model"]
                            )

                            agent.indexer.embed_model = get_embedding_client(
                                embed_info["provider"],
                                embed_info["model"]
                            )

                            # Update service context
                            from llama_index.core import ServiceContext
                            agent.indexer.service_context = ServiceContext.from_defaults(
                                llm=agent.indexer.llm,
                                embed_model=agent.indexer.embed_model
                            )

                            return f"✅ Models updated to: {llm_choice} & {embedding_choice}"
                        except Exception as e:
                            return f"❌ Error updating models: {str(e)}"

                    apply_model_btn.click(
                        update_models,
                        [llm_dropdown, embedding_dropdown],
                        model_status
                    )

                gr.Markdown("## Search Options")

                search_type = gr.Radio(
                    ["hybrid", "graph", "transformed"],
                    label="Retrieval Method",
                    value="hybrid"
                )

                def update_retrieval_type(choice):
                    # This would set the retrieval type on the agent
                    # For demonstration, we'll just return the selected type
                    return f"Using {choice} retrieval"

                search_type.change(update_retrieval_type, search_type, None)

                upload_button = gr.UploadButton(
                    "Upload Document",
                    file_types=[".pdf", ".txt", ".docx"],
                    file_count="multiple"
                )

                def handle_upload(files):
                    file_paths = [file.name for file in files]

                    # Process uploaded files
                    processor = DocumentProcessor()
                    nodes = processor.ingest_documents(file_paths=file_paths)

                    # Update indexes
                    indexer.create_vector_index(nodes, "vector_index_updated")
                    indexer.create_knowledge_graph(nodes, "kg_index_updated")

                    # Update retrievers
                    retriever.vector_retriever = VectorIndexRetriever(
                        index=indexer.vector_index,
                        similarity_top_k=RETRIEVAL_TOP_K
                    )
                    retriever.kg_retriever = KGTableRetriever(
                        index=indexer.kg_index,
                        similarity_top_k=RETRIEVAL_TOP_K
                    )
                    retriever.setup_bm25_retriever(nodes)

                    return f"Uploaded and processed {len(files)} documents"

                upload_button.upload(handle_upload, upload_button, None)

    return interface

# This would be called from the main application
def launch_interface(agent):
    """Launch the interface with the agent."""
    interface = create_interface(agent)
    interface.launch(share=True)