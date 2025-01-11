export interface Game {
    game_id: string;
    matchup: string;
    game_date: string;
    home_team_name: string;
    home_team_abbreviation: string;
    home_team_id: string;
    away_team_name: string;
    away_team_abbreviation: string;
    away_team_id: string;
    winner: string;
    season_id: string;
    video_available: number;
}