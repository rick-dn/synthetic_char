This repo provides a short PoC for generating multiple images synthetic characters where the character is identifiable across each image. 
To achieve this a small dataset  beachybeach for fewshot training was created. Beachybeach is a synthetic character and the dataset has <10 images and corresponding captions.
The dataset was used to finetune a stable diffusion model with LORA. The notebook synth_char_gen contains the corresponding code.

Then, this model was loaded and IP-adapter was used to generate consistent character based on a repo of reference images which are randomly selected. The notebook ip-adapter-sd contains the corresponding code.
As this is a PoC the text prompt has been fixed, enhanncing the text prompt with RAG wil lead to better results. Each time the docker image is run it will produce three images of the same character, showcasing the fact the repo can produce different images of the same syntehtic charatcer. For now there are only limited characters but if you run it multiple times you will get unique images where a single character is identifiable across images. The output is not perfect, but it can be improved with stratagies such as using Flux instead of a small basic stable diffusion model. Enhancing the text prompting through RAG. 

We can run the PoC syntehtic character generation repo with the docker image. the docker image

'''  
docker pull worthlessfella/synthetic-char-infer-cuda12:latest  
docker run --rm --gpus all -v $(pwd)/outputs:/app/outputs --name synthetic-char-infer worthlessfella/synthetic-char-infer-cuda12:latest  
''' 

It stores the output in docker_outputs folder in the present working directory.  

