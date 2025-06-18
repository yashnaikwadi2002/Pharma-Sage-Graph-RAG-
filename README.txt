1. open vscode from target folder
2. In terminal create directory by "mkdir graprag"
3. In terminal create virtual environment "py -3.10 -m venv graphragvenv"

Before deleting use "deactivate" command to deactivate the venv.
Command to Delete the environment : Remove-Item -Recurse -Force .\graphragvenv
again create the new one : python -m venv graphragvenv
Activate : 
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\graphragvenv\Scripts\Activate
Force install the packages: pip install --no-cache-dir -r requirements.txt


4. Add a policy by "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"
5. the activate venv "graphragvenv\Scripts\activate"

6. Install and update the pip "python -m pip install --upgrade pip"
7. Install and ugrade the setuptools package "python -m pip install --upgrade setuptools" 
8. Install GrapgRag library "pip install graphrag"
9. Initialize the Graph Rag "graphrag init --root ."

 Step 1: Create a Google Cloud Project
Go to https://console.cloud.google.com

Click the project dropdown (top-left).

Click "New Project" and give it a name (e.g., graph-rag-project).

Click Create.

Go to link : https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com  To enable GenLang API

Go to link : https://console.cloud.google.com/apis/credentials to create the API key from Credentials.


Go into Neo4j and create the instance and copy the credentials from the downloaded file form the Neo4j.

Go to : https://milvus.io/ for creating milvus account and create a instance for free and copy the credentials from the downloaded file form the Milvus file.

Install required packages.
1. "pip install google-generativeai".
2. "pip install neo4j"
3. "pip install pymilvus"
4. Check the connection of both by making the test.py files and run them to check the connections.


Create the folder structure and create it using the png image in Graphrag folder.
Install the required libraries for langchain. "pip install langchain PyPDF2 pymilvus neo4j fastapi uvicorn streamlit" 

After writing the code in files in Folder Structure then install the required libraries from the requirements.txt by running the command:
			"pip install -r requirements.txt".
There is even a file for testing the Gemini API embeddings and run that .py file check the connection and embedding.

Then next command is "uvicorn main:app --reload" to run the backend API.
Then open the new terminal and run the command "streamlit run frontend/app.py" to run the Streamlit UI.



Optional Changes:
Preloading Example Data (for demo or testing):

If you want your app to work without uploading files every time, I can help you:
Add a load_sample_pdfs() function.
Include some open-source pharma PDFs.
Auto-process and ingest them into Milvus + Neo4j at startup.







dim=768 is required because Gemini’s embedding-001 model returns 768-dimensional vectors.
