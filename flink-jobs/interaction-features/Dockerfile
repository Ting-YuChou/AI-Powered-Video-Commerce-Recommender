FROM maven:3.9-eclipse-temurin-11 AS build

WORKDIR /src
COPY pom.xml .
RUN mvn -q -DskipTests dependency:go-offline

COPY src ./src
RUN mvn -q -DskipTests package

FROM flink:1.19.1-scala_2.12-java11

COPY --from=build /src/target/interaction-features-1.0.0.jar /opt/flink/usrlib/interaction-features.jar
