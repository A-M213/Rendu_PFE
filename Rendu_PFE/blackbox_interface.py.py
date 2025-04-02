import subprocess

def get_path_cost_with_blackbox(seed, path, blackbox_path):
    """
    Appelle la Black Box pour récupérer le coût d'un chemin donné.

    """
    path_str = ",".join(map(str, path))
    try:
        result = subprocess.run([blackbox_path, str(seed), path_str],
                                capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Erreur lors de l'exécution de la Black Box : {e}")
        return None

# Exemple d'utilisation
#blackbox_path = "h:/Desktop/pfe/blackBoxx.exe"
#example = get_path_cost_with_blackbox(0, [17, 14, 39], blackbox_path)
#print("Coût retourné par la Black Box :", example)
