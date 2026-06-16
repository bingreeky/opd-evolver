# FROM mysql
FROM docker.xuanyuan.me/mysql:latest

ENV MYSQL_ROOT_PASSWORD="password"

ADD data/intercode/sql/merged/ic_sql_merged_dbs.sql /docker-entrypoint-initdb.d/