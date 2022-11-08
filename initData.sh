typeset NORMAL_TEMP=$( mktemp )
touch "${NORMAL_TEMP}"
typeset NORMAL_TEMP_LABELLED=$( mktemp )
touch "${NORMAL_TEMP_LABELLED}"

ls cs4243_smallest/normal > "${NORMAL_TEMP}"
awk '{print "normal/"$0",0"}' "${NORMAL_TEMP}" > "${NORMAL_TEMP_LABELLED}"

typeset CARRYING_TEMP=$( mktemp )
touch "${CARRYING_TEMP}"
typeset CARRYING_TEMP_LABELLED=$( mktemp )
touch "${CARRYING_TEMP_LABELLED}"

ls cs4243_smallest/carrying > "${CARRYING_TEMP}"
awk '{print "carrying/"$0",1"}' "${CARRYING_TEMP}" > "${CARRYING_TEMP_LABELLED}"

typeset THREAT_TEMP=$( mktemp )
touch "${THREAT_TEMP}"
typeset THREAT_TEMP_LABELLED=$( mktemp )
touch "${THREAT_TEMP_LABELLED}"

ls cs4243_smallest/threat > "${THREAT_TEMP}"
awk '{print "threat/"$0",2"}' "${THREAT_TEMP}" > "${THREAT_TEMP_LABELLED}"

echo "directory,label" > dataset.csv
cat "${NORMAL_TEMP_LABELLED}" "${CARRYING_TEMP_LABELLED}" "${THREAT_TEMP_LABELLED}" >> dataset.csv
