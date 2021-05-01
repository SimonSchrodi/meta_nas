ifndef data
$(error data is undefined)
endif

ifndef id
$(error id is undefined)
endif

train:
	python3 dev_meta_learner/train_model.py --model ResNet18 --data $(data)

ingest:
	#rm -f "sample_output$(id)"/*
	mkdir "sample_output/$(id)"
	python3 ingestion_program/ingestion.py _ $(data) "sample_output/$(id)" _ _ dev_submission

score:
	#rm -Rf scoring_tmp
	mkdir "scoring_tmp$(id)"
	mkdir "scoring_tmp$(id)"/res
	mkdir "scoring_tmp$(id)"/ref
	cp -r $(data)/* "scoring_tmp$(id)"/ref
	cp "sample_output/$(id)"/* "scoring_tmp$(id)"/res
	python3 scoring_program/score.py "scoring_tmp$(id)" .
	mv scores.txt "scores$(id)".txt
	rm -Rf "scoring_tmp$(id)"

all:
	make ingest data=$(data) id=$(id)
	make score data=$(data) id=$(id)
