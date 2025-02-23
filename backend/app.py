from flask import Flask, jsonify, request
from pyexpat.errors import messages

from SQL.db_manager import init_database
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

        # Query to player stats
        query = """
            SELECT *
            FROM player_stats 
            WHERE "GAME_ID" = %s;
        """
        cursor.execute(query, (game_ID,))
        player_stats = cursor.fetchall()

        # Converts from time stamp to a string
        for player in player_stats:
            # Extract the "MIN" value from the row
            time_delta = player["MIN"]
            # Calculate total seconds
            total_seconds = int(time_delta.total_seconds())

            # Convert to minutes and seconds
            minutes = total_seconds // 60
            seconds = total_seconds % 60

            # Update the player's "MIN" value with the formatted time
            player["MIN"] = f"{minutes:02}:{seconds:02}"


        # Close connection
        cursor.close()
        conn.close()

        # Return the results as JSON
        return jsonify(game_info, box_score, player_stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define a route to set up database for first time user
@app.route('/setup-database', methods=['POST'])
def post_setup_database():
    try:
        init_database()
        # Return the results as JSON
        message = {
            "status": "success",
            "message": "Database initialized successfully",
        }
        return jsonify(message), 200
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
