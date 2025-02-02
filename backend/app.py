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


# Define a route to search for games through our database
@app.route('/box-score', methods=['GET'])
def get_box_score():
    try:
        # Check if variables are set. If not set to default values
        game_id = str(request.args.get('gameID'))


        # Establish database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Query for game info
        query = """
            SELECT "GAME_DATE", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION", "WINNER"
            FROM schedule 
            WHERE "GAME_ID" = %s;
            """
        cursor.execute(query, game_id)
        game_info = cursor.fetchall()

        # Query for box scores
        query = """
            SELECT * 
            FROM team_stats 
            WHERE "GAME_ID" = %s;
        """
        cursor.execute(query, game_id)
        box_score = cursor.fetchall()

        # Close connection
        cursor.close()
        conn.close()

        print(game_info)

        data = {
            'game_info': game_info,
            'box_scores': box_score
        }

        # Return the results as JSON
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
