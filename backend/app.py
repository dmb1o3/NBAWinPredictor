from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
from SQL.config import config_params

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
        # Check if page and limit are set. If not set to default values
        game_ID = str(request.args.get('game_ID'))

        # Establish database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Query to get info about the game played
        query = """
            SELECT "GAME_DATE", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION", "WINNER"
            FROM schedule 
            WHERE "GAME_ID" = %s;
        """
        cursor.execute(query, (game_ID,))
        game_info = cursor.fetchall()

        # Query to get box_scores
        query = """
            SELECT *
            FROM team_stats 
            WHERE "GAME_ID" = %s;
        """
        cursor.execute(query, (game_ID,))
        box_score = cursor.fetchall()


        # Close connection
        cursor.close()
        conn.close()

        # Return the results as JSON
        return jsonify(game_info, box_score)
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
