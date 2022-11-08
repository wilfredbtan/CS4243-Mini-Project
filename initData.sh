echo "File path: $1"

typeset NORMAL_TEMP=$( mktemp )
touch "${NORMAL_TEMP}"
typeset NORMAL_TEMP_LABELLED=$(mktemp)
touch "${NORMAL_TEMP_LABELLED}"

ls "$1/normal" > "${NORMAL_TEMP}"
echo "$1/normal"
awk '{print "normal/"$0",0"}' "${NORMAL_TEMP}" > "${NORMAL_TEMP_LABELLED}"

typeset CARRYING_TEMP=$( mktemp )
touch "${CARRYING_TEMP}"
typeset CARRYING_TEMP_LABELLED=$( mktemp )
touch "${CARRYING_TEMP_LABELLED}"

ls "$1/carrying" > "${CARRYING_TEMP}"
echo "$1/carrying"
awk '{print "carrying/"$0",1"}' "${CARRYING_TEMP}" > "${CARRYING_TEMP_LABELLED}"

typeset THREAT_TEMP=$( mktemp )
touch "${THREAT_TEMP}"
typeset THREAT_TEMP_LABELLED=$( mktemp )
touch "${THREAT_TEMP_LABELLED}"

ls "$1/threat" > "${THREAT_TEMP}"
echo "$1/threat"
awk '{print "threat/"$0",2"}' "${THREAT_TEMP}" > "${THREAT_TEMP_LABELLED}"

echo "directory,label" > dataset.csv
cat "${NORMAL_TEMP_LABELLED}" "${CARRYING_TEMP_LABELLED}" "${THREAT_TEMP_LABELLED}" >> dataset.csv
