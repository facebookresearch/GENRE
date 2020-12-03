default: build dockerize

build:
    mvn clean package -U

dockerize:
    docker build -t git.project-hobbit.eu:4567/gerbil/spotwrapnifws4test .

push:
    docker push git.project-hobbit.eu:4567/gerbil/spotwrapnifws4test