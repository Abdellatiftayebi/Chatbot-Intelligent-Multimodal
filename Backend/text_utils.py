def remove_quotes(text: str) -> str:
    """
    Supprime les guillemets autour d'une chaîne de texte s'ils existent.
    """
    # Supprimer les guillemets doubles ou simples au début et à la fin
    return text.strip('"').strip("'")
