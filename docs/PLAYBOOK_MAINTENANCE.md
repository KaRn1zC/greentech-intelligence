# Playbook de Maintenance - GreenTech Intelligence

>
> Procedures operationnelles pour diagnostiquer et resoudre les incidents en production.

---

## 1. Acces aux outils de monitoring

| Outil | URL locale | Identifiants |
|-------|-----------|-------------|
| **Grafana** | http://localhost:3000 | admin / admin123 |
| **Prometheus** | http://localhost:9090 | - |
| **Loki** | http://localhost:3100 | - |
| **MLflow** | http://localhost:5000 | - |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin123 |
| **API Swagger** | http://localhost:8000/docs | - |

---

## 2. Diagnostiquer un incident via Grafana

### Etape 1 : Identifier le symptome

Ouvrir Grafana > Dashboard **"Performance Systeme"** et verifier :
- **Latence HTTP** : Le graphe montre-t-il un pic au-dessus de 2 secondes ?
- **Taux d'erreurs** : Le compteur d'erreurs 5xx est-il en hausse ?
- **Target UP/DOWN** : L'API, PostgreSQL ou MinIO sont-ils marques comme "down" ?

### Etape 2 : Consulter les logs dans Loki

1. Dans Grafana, aller dans **Explore** > Selectionner la datasource **Loki**
2. Utiliser les requetes LogQL suivantes :

```text
# Tous les logs de l'API (derniere heure)
{container="greentech-api"} |= ""

# Uniquement les erreurs
{container="greentech-api"} |= "ERROR"

# Erreurs de base de donnees
{container="greentech-api"} |= "DatabaseError" or |= "OperationalError"

# Erreurs d'inference IA
{container="greentech-api"} |= "inference" |= "ERROR"

# Logs avec un traceback Python
{container="greentech-api"} |= "Traceback"
```

### Etape 3 : Verifier les metriques Prometheus

Dans Prometheus (http://localhost:9090) > onglet **Graph**, executer :

```promql
# Latence moyenne des requetes (5 min)
rate(greentech_request_duration_seconds_sum[5m]) / rate(greentech_request_duration_seconds_count[5m])

# Nombre de requetes par seconde
rate(greentech_http_responses_total[5m])

# Taux d'erreurs 5xx
rate(greentech_http_responses_total{status=~"5.."}[5m])

# Etat des targets
up
```

---

## 3. Incidents courants et resolution

### Incident : L'API ne repond plus (HTTP 502/503)

**Diagnostic** :
1. Verifier le conteneur : `docker ps | grep greentech-api`
2. Lire les logs : `docker logs greentech-api --tail 50`
3. Verifier la sante : `curl http://localhost:8000/health`

**Causes possibles** :
- **OOM (Out of Memory)** : Le modele IA consomme trop de RAM
  - Solution : Augmenter la limite memoire dans docker-compose ou reduire la taille du batch
- **Port occupe** : Un autre processus utilise le port 8000
  - Solution : `netstat -tlnp | grep 8000` puis tuer le processus
- **Crash du conteneur** : Erreur au demarrage
  - Solution : `docker restart greentech-api`

### Incident : La base de donnees est inaccessible

**Diagnostic** :
1. `docker ps | grep greentech-postgres`
2. `docker exec greentech-postgres pg_isready -U greentech`
3. `docker logs greentech-postgres --tail 30`

**Causes possibles** :
- **Volume corrompu** : `docker volume rm greentech-postgres-data` puis reinitialiser
- **Connexions saturees** : Augmenter `max_connections` dans `postgresql.conf`
- **Disque plein** : Verifier l'espace `docker system df`

### Incident : L'inference IA est tres lente (> 10s)

**Diagnostic** :
1. Dashboard Grafana > "Metier GreenTech" > Temps moyen d'inference
2. Verifier la charge GPU/CPU : `docker stats greentech-api`

**Causes possibles** :
- **Pas de GPU disponible** : Le modele tourne sur CPU (normal en Docker sans GPU passthrough)
  - Solution attendue : Latence 5-15s en CPU est normale pour un modele 3B
- **Modele non charge** : Premiere requete apres demarrage
  - Solution : Le modele est charge en lazy-loading, la premiere inference est plus lente

### Incident : L'API Hugging Face (resume) echoue

**Diagnostic** :
1. Logs : `{container="greentech-api"} |= "summarizer" |= "ERROR"`
2. Verifier le token : La variable `HUGGINGFACE_TOKEN` est-elle definie ?

**Causes possibles** :
- **Rate limit** : L'API gratuite a des limites
  - Solution : Attendre ou passer a un plan payant
- **Token expire** : Renouveler le token sur huggingface.co
- **API indisponible** : Verifier https://status.huggingface.co

---

## 4. Commandes utiles

```bash
# Etat de tous les conteneurs
docker compose ps

# Logs d'un service specifique (temps reel)
docker compose logs -f api

# Redemarrer un service sans toucher aux autres
docker compose restart api

# Reconstruire et relancer un service
docker compose up -d --build api

# Nettoyer les images inutilisees
docker system prune -f

# Verifier l'espace disque Docker
docker system df

# Executer une commande dans le conteneur API
docker exec -it greentech-api python -c "from greentech.ai.models.inference import get_classifier; print(get_classifier())"

# Backup de la base PostgreSQL
docker exec greentech-postgres pg_dump -U greentech greentech_db > backup_$(date +%Y%m%d).sql

# Restaurer un backup
cat backup_20260411.sql | docker exec -i greentech-postgres psql -U greentech greentech_db
```

---

## 5. Contacts et escalade

| Niveau | Action | Responsable |
|--------|--------|-------------|
| N1 | Lecture des logs Grafana, redemarrage conteneurs | KaRn1zC |
| N2 | Analyse approfondie, correction de code | KaRn1zC |
| N3 | Infrastructure Render, incidents cloud | Support Render |

