# from django.test import TestCase
# import openai
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from dbapp.models import SummarizationResult
# from dbapp.serializers import SummarizationResultSerializer
# from dbapp.text_utils import CharacterTextSplitter
# from dbapp.models import *
# from langchain import LangChain



# class SummarizeAPIView(APIView):
#     def post(self, request, *args, **kwargs):
#         try:
#             file = request.FILES.get('file')

#             if not file:
#                 return Response({"error": "File not provided"}, status=status.HTTP_400_BAD_REQUEST)

#             text_content = file.read().decode('utf-8')

#             text_splitter = CharacterTextSplitter()
#             texts = text_splitter.split_text(text_content)

#             openai_api_key = "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"

#             # Initialize LangChain with desired parameters
#             lang_chain = LangChain(
#                 api_key=openai_api_key,
#                 model="text-davinci-003",
#                 temperature=0.7,
#                 max_tokens=100,  # Adjust as needed
#             )

#             # Generate summaries for the text using LangChain
#             summaries = lang_chain.generate_summaries(texts)

#             # Combine the summarized paragraphs into a single text
#             summarized_text = "\n".join(summaries)

#             # Save the summarized text to the database
#             summarization_result = SummarizationResult.objects.create(summary=summarized_text)

#             # Serialize the result and return it in the response
#             serializer = SummarizationResultSerializer(summarization_result)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Create your tests here.









# pdf_summarizer/views.py
import os
import tiktoken
import gradio as gr
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import PDFDocument
from .serializers import PDFDocumentSerializer
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

os.environ["OPENAI_API_KEY"] = "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"

class PDFSummarizer(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        pdf_file = request.data.get('file')
        if not pdf_file:
            return Response({'error': 'File is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Save the uploaded PDF file
            pdf_document = PDFDocument(file=pdf_file)
            pdf_document.save()

            # Summarize the uploaded PDF
            pdf_file_path = pdf_document.file.path
            llm = OpenAI(temperature=0)
            loader = PyPDFLoader(pdf_file_path)
            docs = loader.load_and_split()
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(docs)

            return Response({'summary': summary}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                                                                                                                                                                                                                                                                                                                                                                                                                                                        








# Import necessary modules
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import TextDocument  # Import the TextDocument model
import os
import tempfile
from django.core.files.uploadedfile import UploadedFile
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from django.conf import settings
from django.db import transaction
import textract

# Replace with your OpenAI API key
openai_api_key = "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"

GPT_ENGINE = "gpt-3.5-turbo"

class SummarizeTextView(APIView):
    parser_class = (FileUploadParser,)  # Enable file uploads

    @transaction.atomic
    def post(self, request, format=None):
        try:
            # Get the uploaded file
            uploaded_file: UploadedFile = request.FILES['file']
            file_name, file_extension = os.path.splitext(uploaded_file.name)

            # Create a temporary file to store the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)

            # Read the content of the uploaded file
            if file_extension.lower() == '.pdf':
                # Extract text from a PDF file
                file_content = textract.process(temp_file.name)
                file_content = file_content.decode('utf-8')
            elif file_extension.lower() == '.doc':
                # Extract text from a DOC file
                file_content = textract.process(temp_file.name, extension='docx')
                file_content = file_content.decode('utf-8')
            else:
                # For text files, read directly from the temporary file
                with open(temp_file.name, 'r') as text_file:
                    file_content = text_file.read()

            # Split the input text into segments of maximum length 4096 tokens
            max_tokens = 4096
            segments = [file_content[i:i+max_tokens] for i in range(0, len(file_content), max_tokens)]

            # Initialize an empty summary
            summary = ""

            # Function to generate a summary using GPT-3.5-turbo
            def generate_summary(text):
                openai.api_key = openai_api_key
                response = openai.ChatCompletion.create(
                    model=GPT_ENGINE,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": text},
                    ],
                )
                return response.choices[0].message["content"]

            # Generate a summary for each segment and concatenate them
            for segment in segments:
                segment_summary = generate_summary(segment)
                summary += segment_summary

            # Save the original text and summary to the database
            text_document = TextDocument(original_text=file_content, summary=summary)
            text_document.save()

            # Remove the temporary file
            os.remove(temp_file.name)

            return Response({'summary': summary, 'document_id': text_document.id}, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# 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





# import os
# import gradio as gr
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from rest_framework import status
# from django.core.files.uploadedfile import SimpleUploadedFile
# from langchain import OpenAI
# from langchain.chains.summarize import load_summarize_chain
# from langchain.document_loaders import TextLoader
# from .models import TextDocument  # Import the TextDocument model and the serializer


# os.environ["OPENAI_API_KEY"] = "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"

# class TextSummarizer(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, format=None):
#         text_file = request.data.get('file')

#         # if not text_file:
#         #     return Response({'error': 'File is required.'}, status=status.HTTP_400_BAD_REQUEST)

#         # try:
#             # Create a SimpleUploadedFile from the uploaded file
#         uploaded_file = SimpleUploadedFile(text_file.name, text_file.read())

#         # Read the text content
#         text_content = uploaded_file.read()

#         # Save the text content to the database
#         text_document = TextDocument(content=text_content)
#         text_document.save()

#         # Summarize the uploaded text content
#         llm = OpenAI(temperature=0)
#         loader = TextLoader(text_content)
#         docs = loader.load_and_split()
#         chain = load_summarize_chain(llm, chain_type="map_reduce")
#         summary = chain.run(docs)

#         # Close and remove the temporary file
#         uploaded_file.close()

#         return Response({'summary': summary}, status=status.HTTP_200_OK)
#         # except Exception as e:
#         #     return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




# from django.test import TestCase
# import openai
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from api.models import SummarizationResult
# from api.serializers import PDFDocumentSerializer
# from api.text_utils import CharacterTextSplitter

# class SummarizeAPIView(APIView):
#     def post(self, request, *args, **kwargs):
#     # try:
#         file = request.FILES.get('file')

#         if not file:
#             return Response({"error": "File not provided"}, status=status.HTTP_400_BAD_REQUEST)

#         text_content = file.read().decode('utf-8')

#         text_splitter = CharacterTextSplitter()
#         texts = text_splitter.split_text(text_content)

#         openai_api_key = "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"

#         # Initialize OpenAI with your API key
#         openai.api_key = openai_api_key

#         # Join the text paragraphs into a single text
#         combined_text = "\n".join(texts)

#         # Use OpenAI to summarize the combined text
#         response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=combined_text,
#             max_tokens=100,  # Adjust as needed
#             temperature=0.7
#         )

#         summarized_text = response.choices[0].text

#         # Save the summarized text to the database
#         summarization_result = SummarizationResult.objects.create(summary=summarized_text)

#         # Serialize the result and return it in the response
#         serializer = PDFDocumentSerializer(summarization_result)
#         return Response(serializer.data, status=status.HTTP_201_CREATED)
        # except Exception as e:
        #     return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
