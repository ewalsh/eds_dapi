# eds_dapi
Demo App For EDS Analytics &amp; Publications

# to get started
`docker-compose up --build`
- Just noting after the initial build, you can use just `docker-compose start` and `docker-compose stop`

# zeppelin link
http://localhost:8087

# updates to zeppelin and django
1. create a folder at the root named `zdata`
  - I have added to the docker-compose to share data with the container
  - I have also added seaborn and statsmodels
2. To make sure you have the latest docker images please run:
  - `docker pull ewalsh200/python-nginx`
  - `docker pull ewalsh200/zeppelin:focal`
