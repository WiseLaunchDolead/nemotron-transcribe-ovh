# Nemotron Transcribe sur OVHcloud

API de transcription française avec horodatage des mots et identification des
intervenants. Construction dans GitHub Actions, sans Docker local.
**L'inférence et la mémoire sur L40S restent à mesurer.**

## Implémentation

- Runtime officiel [NVIDIA/NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp/tree/20999cc921d2b14aa58708f61756a8b2e787ba20), commit et sous-modules figés.
- [Paire AtomicChat](https://huggingface.co/AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF/tree/dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59) : **AD-IQ4_NL**, 19 678 656 672 octets, et encodeur F16, 1 670 410 080 octets. Total ≈21,35 Go / 19,88 Gio sur disque, hors mémoire d'exécution. Révision, tailles et SHA-256 : `models.lock.json`.
- CUDA 12.8.1, Ubuntu 24.04, bases Docker figées par digest, architecture CUDA **89** pour L40S. Vérifier la compatibilité du pilote OVH avant lancement.
- Serveur HTTP NVIDIA natif ; pas de FastAPI. Sous-ensemble transcription compatible avec les champs OpenAI documentés ; diarisation = extension NVIDIA.
- Profil prudent : 4 séquences, 32 768 cellules de contexte partagé, MTP désactivé. Environ **29 minutes maximum** par audio selon le budget de texte ; commencer avec 20 à 60 secondes. Le support d'une heure du modèle n'est pas activé dans ce profil.
- L'encodeur SALM contient déjà la branche locuteurs et la tête CTC ; pas de Sortformer séparé.

Références : [SALM](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/20999cc921d2b14aa58708f61756a8b2e787ba20/docs/salm/README.md), [mémoire](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/20999cc921d2b14aa58708f61756a8b2e787ba20/docs/salm/configuration.md), [API](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/20999cc921d2b14aa58708f61756a8b2e787ba20/docs/api.md).

## Build cloud et registre

Un push sur `feat/cloud-build-scaffold` exécute les tests, compile `linux/amd64`,
vérifie les bibliothèques et l'UID 42420, puis publie :

```text
ghcr.io/wiselaunchdolead/nemotron-transcribe-ovh:sha-<SHA complet du commit>
ghcr.io/wiselaunchdolead/nemotron-transcribe-ovh:candidate
```

Sur `main`, l'alias est `:main`. Les PR exécutent les tests Python ; la compilation
CUDA s'exécute sur push et déclenchement manuel. Aucun GPU OVH ni poids n'est utilisé
par ce workflow. Déployer le **digest immuable** `ghcr.io/...@sha256:...` vérifié.

GHCR peut créer le package en privé même si le dépôt est public. Dans GitHub :
profil → Packages → package → Package settings → Change visibility → Public,
ou configurer les identifiants du registre privé dans OVH. Vérifier l'accès anonyme
avant de sélectionner une image publique. Aucun secret HF/OVH dans GitHub Actions.

## Configuration OVH

| Paramètre | Valeur |
|---|---|
| Projet et région | Kalepin, GRA (Gravelines) |
| Image | digest GHCR ayant passé les vérifications |
| GPU | 1 × L40S, 45 Gio affichés dans la passation |
| Port | 8080 |
| Accès | **Restreint** |
| Variable | `AUTH_MODE=ovh` |
| Label | `app=nemotron` (valeur limitée à 15 caractères dans la console) |
| Readiness probe | `/ready`, port 8080 |
| Scaling après validation | min 0, max 1 |
| Modèles | au moins 25 Go libres à `/workspace/models` |

En mode `AUTH_MODE=ovh`, l'entrée OVH assure l'authentification. Ne pas définir
`NEMO_SPEECH_HTTP_API_KEY` : deux clés Bearer différentes utiliseraient le même
en-tête `Authorization`. Créer un jeton AI régional GRA limité au label et au rôle
minimal permettant les appels. **Ne jamais rendre publique une app dans ce mode.**
Les probes internes n'ont pas besoin de jeton.

Si la console ne propose pas de champ de variable d'environnement, configurer la
commande Docker `env AUTH_MODE=ovh /opt/app/start.sh` et vérifier sa prise en compte
dans l'équivalent CLI avant de commander.

Hors accès restreint OVH, utiliser le mode par défaut `AUTH_MODE=api-key` avec le
secret `NEMO_SPEECH_HTTP_API_KEY`. Le démarrage sans clé est alors refusé.
`--check` vérifie les binaires sans serveur, GPU ou téléchargement.
Sur un runner CPU, cette vérification utilise la cible Docker `verify`, qui contient
le stub du pilote CUDA officiel dans un chemin réservé aux tests. Cette cible n'est
jamais publiée ; la cible `runtime` utilise le vrai pilote fourni par OVH.

Le démarrage vérifie la présence du GPU, télécharge la paire publique figée,
reprend les téléchargements interrompus et vérifie tailles et SHA-256 avant de
publier les fichiers dans le cache. Un cache existant corrompu provoque un arrêt
explicite. `MODEL_DIR` permet de choisir un volume monté.

Le cache local est **éphémère** ; sa conservation après passage à zéro n'est pas
garantie. Pour des réveils réguliers, précharger la paire vérifiée dans un bucket
Object Storage GRA, puis monter ce bucket avec cache OVH. Le stockage est facturé
séparément. Prévoir un montage en lecture/écriture accessible à l'UID 42420 : le
téléchargeur crée un fichier de verrou même lorsque les poids sont préchargés.
Un premier essai sur disque éphémère permettra de mesurer le coût à froid.

La readiness arrive après téléchargement, chargement et warmup. Le `HEALTHCHECK`
Docker prévoit 30 minutes de grâce ; cela ne configure pas les probes OVH. Adapter
leur tolérance aux mesures et consulter les logs pendant l'initialisation.

## Test français à deux voix

Préparer un WAV de 20–60 secondes contenant deux personnes et sa transcription de
référence. Conversion possible avec FFmpeg :

```bash
ffmpeg -i conversation.mp3 -ar 16000 -ac 1 -c:a pcm_s16le conversation.wav
```

Définir `AI_APP_TOKEN` via le gestionnaire de secrets du client, puis :

```bash
python3 scripts/smoke_test.py \
  --url https://<app-id>.app.gra.ai.cloud.ovh.net \
  --audio conversation.wav --min-speakers 2 --output results/first-run
```

Le test vérifie rejet sans jeton, readiness et capacités, puis envoie **une seule**
transcription. Il conserve `transcription.json` et `metrics.json` hors Git et échoue
si texte, horodatages ou deux locuteurs manquent. Comparer humainement à la référence
pour vérifier qualité linguistique et attribution réelle des voix.

Pour l'application ou n8n :

```text
POST <URL>/v1/audio/transcriptions
Authorization: Bearer <jeton OVH>
Content-Type: multipart/form-data (généré par le client)

file = WAV binaire
language = fr
response_format = verbose_json
diarization = true
```

Dans n8n : HTTP Request, POST, credential Bearer, Form-Data, `file` de type n8n
Binary File et les autres champs texte. Prévoir un timeout adapté au réveil ; pas
de retry automatique durant la validation. La réponse contient `text`, `duration`
et `words[]` (`word`, `start`, `end` en secondes, `speaker` entier à partir de 1).
Le runtime donne des horodatages de **mots**, pas de segments natifs : regrouper côté
client par locuteur et pause si nécessaire.

## Limites

- La diarisation change le prompt : le texte peut perdre ponctuation et majuscules. Pour un prompt personnalisé, voir l'API chat dans le guide SALM.
- Un alignement raté peut rendre le texte sans mots horodatés ; le test échoue dans ce cas.
- Le serveur NVIDIA est présenté comme destiné à l'intégration directe. Valider charge, isolation et erreurs avant production.
- Mesurer VRAM après chargement et pic pendant l'inférence dans Grafana OVH. Le poids disque ne suffit pas à valider L40S.
- Les tests CI n'effectuent aucune inférence. Un build vert ne valide pas le GPU.

Suivre [le protocole de validation](docs/validation-ovh.md). Le tarif 1,55 € HT/h et
le crédit d'essai proviennent de la passation : les confirmer dans la console.
OVH facture les phases `SCALING` et `RUNNING` à la minute ; le démarrage ne doit pas
être supposé gratuit.

## Tests sans GPU

```bash
python3 -m unittest discover -s tests -v
bash -n start.sh
```

Modèle : [OpenMDW-1.1](https://huggingface.co/AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF/blob/dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59/LICENSE).
Les notices runtime sont conservées dans `/opt/nemo/share`.

Sources OVH : [UID et image](https://support.us.ovhcloud.com/hc/en-us/articles/37138294111251-AI-Deploy-Tutorial-Build-use-a-custom-Docker-image),
[accès](https://docs.ovhcloud.com/en/guides/public-cloud/ai-machine-learning/ai-deploy-getting-started),
[autoscaling](https://docs.ovhcloud.com/en/guides/public-cloud/ai-machine-learning/ai-deploy-apps-deployments),
[facturation](https://docs.ovhcloud.com/fr/guides/public-cloud/ai-machine-learning/ai-deploy-billing).
