FROM maven:3.9-eclipse-temurin-17 AS build

WORKDIR /src
COPY pom.xml .
RUN mvn -q -DskipTests dependency:go-offline
RUN mvn -q -DskipTests \
    -DincludeArtifactIds=hadoop-client-api,hadoop-client-runtime \
    -DoutputDirectory=/src/hadoop-libs \
    dependency:copy-dependencies

COPY src ./src
RUN mvn -q -DskipTests package

FROM flink:1.20.1-scala_2.12-java17

COPY --from=build /src/hadoop-libs/ /opt/flink/lib/
COPY --from=build /src/target/interaction-features-1.0.0.jar /opt/flink/usrlib/interaction-features.jar
