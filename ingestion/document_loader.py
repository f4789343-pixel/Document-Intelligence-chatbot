from langchain_community.document_loaders import TextLoader, DirectoryLoader

loader = DirectoryLoader("Documents/", glob="*.text", loader_cls=TextLoader, loader_kwargs={"encoding":"utf-8"})

documents = loader.load()
for doc in documents:
 print(doc)