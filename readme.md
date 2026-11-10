### Documentation-intelligence-chatbot-Project

## Project workflow

* User will upload the files that need to analyze.
* Ask questions about the documents.
* chatbot will answer the questions based on the documents that user provided.

## implementation
* Load the documents using langchain TextLoader
* created splits for each document using the longchain RecursiveCharcterTextsplitter
* converted the splits into embeddings(numerical vectors) using langchain HuggingfaceEmbeddings
* Stored the embeddings into vector space using chroma
* Use the stored vectors to search meanings in data, which retriever
* To search based on user's question keywords, used BM25 retriever.
* To search meaning, patterns with similar keywords search, I used hybrid search.
* To re-evaluate the similar chunks, used CrossEncoder.
* After finding the similar context based on the user's question.
* Send the context and query to the LLM model, which is from chatOpenRouter.
* It will return user interactive response using those chunks.
* Connected the fastapi endpoints.
* Used streamlit for web based interaction.

## What I learned
* I have built this model from scratch.
* Learned how to use Rag syntax in real model.
* Learned how to use fast api upload files and chat endpoints.
* Learned the how to use streamlit while building.
* Learned how to connect streamlit with fast api endpoints.
* Learned why time optimization is important and how to make computation faster.




