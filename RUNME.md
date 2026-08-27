# RUNME

## Starting
 - First, we would run a git clone : git clone -b main https://github.com/TanmayJDesai/panini-course-project-Group-Tanmay-Ethan-Denny.git /content/panini-course-project
 - Open the Panini_Course_Project.ipynb in google colab (File --> Upload Notebook --> Find the Panini_Course_Project.ipynb)
 - Set the runtime type (while T4 works, we highly recommend A100 as it is faster for the full runs)
 - Edit the config cell for whatever pass you are running and run it by clicking Runtime --> Run All. 

## Various Passes (run them in order where only one flag changes to true at a time)
 - So, first set the Question_limit to 2 and all the RUN flags to flase which confirms the setup.
 It is important to note that these jsonl files I'm going to mention will be google drive intermediate cache files and these will jsut be used in the materialize submission section that I used to make the submission files. 
 - Then we run the decomposition step by setting the run_decomposition to true and chaging question limit to none. 
   - This will just populate decompositions.jsonl for all 100 questions per ds. 
 - Then you would change the run_rerank... to true and decomposition to false.
   - This will populate the ricr_trances.jsonl for all 100 questions per ds. 
 - Then you would change the run_answer... to true and rerank to false. 
   - This would populate answers.jsonl for all 100 questions per ds
 - Then you would change the run_ablations to true and run_answer... to false. 
   - This would populate ablation_traces.ksonl and ablation_answers.jsonl with 10 configs x the fixed 20 querstion slice per ds. 
 - Finally with everything set to false in the config tab and question limit on none, run the question 12 cells to use the previously made and cached jsonl files to materialize and get the four submission files and environments.txt

## Seeds
In question 3, when we do sampling, it uses the seed=232. That is the only seed needed. 


## Expected runtime (As I mentioned I did it on A100 gpu, but these times will vary by GPU and cace state)
- Decomposition has 200 questions and it took me about 45 minutes.
- Ricr step has 200 questions and single config and it took me about 1 hour and 15 minutes.
- Answering has 200 questions and it took me about 45 minutes.
- Ablations has 10 configs with 40 questions and this was my longest one, it took me about 3 hours and 20 minutes (this was where I realized T4 was not good and bought A100) 

## Restart points
The jsonl files I mentioned above are all checkpoints all cached and every stage checkpoints to these jsonl by question id after each of the questions finish. So, if you do need to reconnect (happened twice for me), it will just skip the completed question IDs and not rerun them.

## Final commit
fb03e3930660e8873cb59fcc688675179f936816