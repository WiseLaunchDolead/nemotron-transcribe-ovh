# Validation L40S : protocole et mesures

Ce document est un protocole, pas un résultat de benchmark.

## Avant lancement

1. Vérifier build vert, accès GHCR, digest `linux/amd64`, crédit et tarif actuels.
2. Vérifier 1 L40S maximum, région GRA, pilote compatible CUDA 12.8 et quota disponible.
3. Accès restreint, `AUTH_MODE=ovh`, jeton limité au label, port 8080, probe `/ready`.
4. Préparer un WAV français à deux voix (20–60 s), sa référence et le script de test.
5. Préparer le bouton d'arrêt. Réserver un créneau surveillé de 30 minutes, plafond initial d'une heure GPU. À 1,55 €/h indicatif : ≈0,78 € / 30 min, 1,55 € / heure, hors options et arrondis. **Ce plafond est manuel : aucun arrêt automatique n'est configuré.**

## Premier essai surveillé

1. Démarrer un seul réplica ; noter début `SCALING`, fin du téléchargement et readiness.
2. Consulter les logs : GPU détecté, poids vérifiés, modèle complet chargé sans OOM.
3. Exécuter `scripts/smoke_test.py` une fois ; comparer texte, horodatages et voix à la référence.
4. Relever dans Grafana VRAM après chargement et pic d'inférence, RAM, latence.
5. Arrêter en cas d'OOM, pilote incompatible ou démarrage bloqué ; diagnostiquer avant nouvel essai.

## Autoscaling 0 → 1 → 0

1. Minimum 0, maximum 1. Commencer par une métrique CPU : la RAM garde les poids et peut empêcher la réduction basée sur son seuil.
2. Relever les deux délais, réduction et passage à zéro. OVH indique qu'ils s'additionnent : un seul champ à 300 s ne garantit pas 300 s au total.
3. Après la dernière requête, ne plus sonder l'URL publique. Observer depuis la console/API de contrôle OVH pour ne pas maintenir ou réveiller l'app.
4. Constater `STANDBY` et zéro réplica. Envoyer une seule requête de readiness authentifiée ; relever temps de réveil et erreurs éventuelles.
5. Vérifier si les poids sont retéléchargés. Refaire ensuite une transcription à chaud ; ne pas relancer aveuglément une transcription expirée.
6. Observer le retour à zéro et la consommation, puis arrêter l'app après le créneau. `STANDBY` reste réveillable ; `STOPPED` clôt l'essai.

Un GPU n'est pas réservé après passage à zéro ; le réveil peut échouer si le flavor
n'est plus disponible. Ajuster client et cache aux mesures observées.

## Relevé à compléter

| Observation | Résultat |
|---|---|
| Date, digest image | À mesurer |
| GPU, VRAM totale, pilote | À mesurer |
| SHA-256 décodeur et encodeur | `models.lock.json` |
| Téléchargement à froid / temps avant readiness | À mesurer |
| VRAM après chargement / pic d'inférence | À mesurer |
| Durée audio / latence à chaud | À mesurer |
| Texte et attribution des deux voix | À vérifier humainement |
| Cohérence des horodatages à l'écoute | À vérifier humainement |
| Temps de passage à zéro / réveil | À mesurer |
| Cache conservé ou retéléchargé | À observer |
| GPU disponible au réveil | À observer |
| Minutes GPU, stockage, coût HT | À vérifier dans OVH |

Sources : [scaling](https://docs.ovhcloud.com/en/guides/public-cloud/ai-machine-learning/ai-deploy-apps-deployments),
[facturation](https://docs.ovhcloud.com/fr/guides/public-cloud/ai-machine-learning/ai-deploy-billing).
