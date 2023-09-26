from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Set your OpenAI API key here as a constant
OPENAI_API_KEY = "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"

class SummarizeView(APIView):
    def post(self, request, format=None):
        try:
            # You can use the predefined API key here
            openai_api_key = OPENAI_API_KEY
            text_input_file = request.FILES.get('text_input')  # Assuming the file field is named 'text_input'

            # Read the content of the uploaded file
            text_input = text_input_file.read().decode('utf-8')

            # Instantiate the LLM model
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

            # Split text
            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(text_input)

            # Create multiple documents
            docs = [Document(page_content=t) for t in texts]

            # Text summarization
            chain = load_summarize_chain(llm, chain_type='map_reduce')
            response = chain.run(docs)

            data = {'summary': response}
            return Response(data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
