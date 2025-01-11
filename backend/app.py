from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
from SQL.config import config_params
import os

app = Flask(__name__)
CORS(app)

# Database connection settings
DB_CONFIG = config_params


# Connect to PostgreSQL
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


# Define a route to get the most recent games
@app.route('/recent-games', methods=['GET'])
def get_recent_games():
    try:
        # Check if page and limit are set. If not set to default values
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        # Calculate offset for proper querying of database
        offset = (page - 1) * limit

        # Establish database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Query for most recent games
        query = """
            SELECT * 
            FROM schedule 
            ORDER BY "GAME_DATE" DESC 
            LIMIT %s OFFSET %s;
        """
        cursor.execute(query, (limit, offset))
        games = cursor.fetchall()

        # Close connection
        cursor.close()
        conn.close()

        # Return the results as JSON
        return jsonify(games)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
