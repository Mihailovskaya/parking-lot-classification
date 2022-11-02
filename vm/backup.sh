BACKUP_DIRECTORY="/media/disk/backups"
MAX_FILE_COUNT=100


FILES_COUNT="$(ls $BACKUP_DIRECTOY | wc -l)"


docker exec -t -u postgres parking pg_dumpall | egrep -v '^CREATE ROLE postgres;' > "$BACKUP_DIRECTORY/dump_`date +%d-%m-%Y"_"%H_%M_%S`.sql"


if [ "$FILES_COUNT" -eq "$MAX_FILE_COUNT" ]; then
    rm "$BACKUP_DIRECTORY/$(ls -t $BACKUP_DIRECTORY| tail -1)"

fi