Required: `docker`, `nvidia-docker2`

**All commands must be run from parent (geology) folder** 

## Building docker image

*tag* - is name for docker image 

To build docker image run the command:

    docker build -t <tag> Dockerfile .

## Running docker container

*run_name* - is name for container born by image

#### For docker `version < 19.03`:

Run docker container without jupyter notebook access and no shared folders using:
* `nvidia-docker run -it --rm --name=<run_name> <tag>`

Run docker container without jupyter notebook access but share a folder:
* `nvidia-docker run -v <host dir>:<docker dir> -it --rm --name=run_name <tag>`

Run docker container with jupyter notebook access and shared folder:
* `nvidia-docker run -p <host port>:8888 -v <host dir>:<docker dir> -it --rm --name=run_name <tag>`


#### For docker `version >= 19.03`:
Run docker container without jupyter notebook access and no shared folders using:
* `docker run --gpus=all -it --rm --name=<run_name> <tag>`

Run docker container without jupyter notebook access but share a folder:
* `docker run --gpus=all -v <host dir>:<docker dir> -it --rm --name=run_name <tag>`

Run docker container with jupyter notebook access and shared folder:
* `docker run --gpus=all -p <host port>:8888 -v <host dir>:<docker dir> -it --rm --name=run_name <tag>`

## Running Jupyter Notebook

To run jupyter notebook inside a container run:
    
    jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root
