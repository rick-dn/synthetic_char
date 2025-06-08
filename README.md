We can run the PoC syntehtic character generation repo with the docker image.

'''
docker pull worthlessfella/synthetic-char-infer-cuda12:latest  
docker run --rm --gpus all -v $(pwd)/outputs:/app/outputs --name synthetic-char-infer worthlessfella/synthetic-char-infer-cuda12:latest  
It stores the output in docker_outputs folder in the present working directory.  


Each time the docker image is run it will produce three images of the same character, showcasing the fact the repo can produce different images of the same syntehtic charatcer.
For now there are only limited characters but if you run it multiple times you will get unique images where a single character is identifiable across images.
The output is not perfect, it can be better, it is just a small pilot project.
