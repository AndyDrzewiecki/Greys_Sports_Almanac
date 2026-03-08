#!/usr/bin/env bash
# ============================================================
# Mini PC (Ubuntu) — PostgreSQL setup for Market Watch Agents
# Run as: sudo bash mini_pc_postgresql.sh
# ============================================================
set -euo pipefail

DB_NAME="market_watch"
DB_USER="mwagent"
DB_PASS="changeme_before_running"   # <-- change this
MWA_HOST_IP="0.0.0.0"              # set to Mini PC's LAN IP to restrict access

echo "=== Installing PostgreSQL ==="
apt-get update -q
apt-get install -y postgresql postgresql-contrib

echo "=== Starting PostgreSQL service ==="
systemctl enable postgresql
systemctl start postgresql

echo "=== Creating database and user ==="
sudo -u postgres psql <<SQL
CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';
CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
SQL

echo "=== Configuring remote access ==="
PG_VERSION=$(pg_lsclusters | awk 'NR==2 {print $1}')
PG_CONF="/etc/postgresql/${PG_VERSION}/main/postgresql.conf"
PG_HBA="/etc/postgresql/${PG_VERSION}/main/pg_hba.conf"

# Allow connections from the local LAN subnet (adjust as needed)
echo "host  ${DB_NAME}  ${DB_USER}  192.168.1.0/24  scram-sha-256" >> "${PG_HBA}"
sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" "${PG_CONF}"

systemctl restart postgresql

echo "=== Installing Python dependencies for the ZBook ==="
echo "  On the ZBook, run: pip install psycopg2-binary"

echo ""
echo "=== DONE ==="
echo "Add this to mwa/.env on all machines:"
echo ""
echo "DB_URL=postgresql://${DB_USER}:${DB_PASS}@<MINI_PC_LAN_IP>/${DB_NAME}"
echo ""
echo "Then run: python -c \"from storage.database import init_db; init_db()\" to create tables."
