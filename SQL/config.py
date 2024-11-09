config_params = {
    "dbname":"nba",
    "user":"postgres",  # Change to personal info
    "password":"YOU_PASSWORD_HERE", # Change to personal info
    "host":"localhost",  # Change if not local
    "port":"5432",  # Change if not
}


config_params_no_db = {
    "user": config_params["user"],  # Change to personal info
    "password": config_params["password"], # Change to personal info
    "host": config_params["host"],  # Change if not local
    "port": config_params["port"],  # Change if not
}

conn_string = ("postgresql://" + config_params["user"] + ":" + config_params["password"] + "@" +
               config_params["host"] + ":" + config_params["port"] + "/" + config_params["dbname"])

