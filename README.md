# Machine Learning Task - TouchNote

The solution consists of a bert-based model wrapped in a FastApi server, made availabe to user for predictions via REST endpoint. 
The above mentioned API is dockerized for easy and fast testing and deployment.

Steps to run the API:
1. Clone the project.
2. Run 'docker-compose up' command.
3. After installation of all dependencies, the uvicorn server will start running on http://localhost:8000
4. The API docs can be access on http://localhost:8000/docs, you will find the sample request body and endpoints.
5. The predict API has the following url, http://localhost:8000/predict
