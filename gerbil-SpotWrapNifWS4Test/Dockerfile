FROM tomcat:8.0.36-jre8

# remove the default tomcat application
RUN rm -rf /usr/local/tomcat/webapps/ROOT /usr/local/tomcat/webapps/ROOT.war

ADD target/gerbil-spotWrapNifWS4Test-0.0.1-SNAPSHOT.war /usr/local/tomcat/webapps/ROOT.war

