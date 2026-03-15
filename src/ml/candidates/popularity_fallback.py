from src.core.interactions_table import InteractionMatrix

def get_popular_candidates(conf):
    interaction_matrix = InteractionMatrix()
    interaction_matrix.update_matrix()
    top_users_df = interaction_matrix.get_popularity_scores(top_k=10)
    return top_users_df