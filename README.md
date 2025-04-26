# cs224n-assignment-5-self-attention-transformers-and-pretraining-q1-a-b-c-d-solved
**TO GET THIS SOLUTION VISIT:** [CS224N: Assignment 5: Self-Attention, Transformers, and Pretraining Q1 a, b, c, d Solved](https://www.ankitcodinghub.com/product/cs224n-assignment-5-self-attention-transformers-and-pretraining-q1-a-b-c-d-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;81427&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;4&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (4 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS224N: Assignment 5: Self-Attention, Transformers, and Pretraining  Q1 a, b, c, d Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (4 votes)    </div>
    </div>
<table width="623">
<tbody>
<tr>
<td width="623"><strong>Note. </strong>Here are some things to keep in mind as you plan your time for this assignment.

•&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; There are math questions again!

•&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The total amount of PyTorch code to write, and code complexity, of this assignment is lower than Assignment 4. However, you’re also given less guidance or scaffolding in how to write the code.

•&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This assignment involves a pretraining step that takes approximately 2 hours to perform on Azure, and you’ll have to do it twice.
</td>
</tr>
</tbody>
</table>
This assignment is an investigation into Transformer self-attention building blocks, and the effects of pretraining. It covers mathematical properties of Transformers and self-attention through written questions. Further, you’ll get experience with practical system-building through repurposing an existing codebase. The assignment is split into a written (mathematical) part and a coding part, with its own written questions. Here’s a quick summary:

<ol>
<li><strong>Mathematical exploration: </strong>What kinds of operations can self-attention easily implement? Why should we use fancier things like multi-headed self-attention? This section will use some mathematical investigations to illuminate a few of the motivations of self-attention and Transformer networks. <strong>Note: </strong>for all questions, you should justify your answer with mathematical reasoning when required.</li>
<li><strong>Extending a research codebase: </strong>In this portion of the assignment, you’ll get some experience and intuition for a cutting-edge research topic in NLP: teaching NLP models facts about the world through pretraining, and accessing that knowledge through finetuning. You’ll train a Transformer model to attempt to answer simple questions of the form “Where was person [x] born?” – without providing any input text from which to draw the answer. You’ll find that models are able to learn some facts about where people were born through pretraining, and access that information during fine-tuning to answer the questions.</li>
</ol>
Then, you’ll take a harder look at the system you built, and reason about the implications and concerns about relying on such implicit pretrained knowledge.

This assignment was originally created by John Hewitt, CS 224N Head TA in Winter 2021.

1

<h1>1.&nbsp;&nbsp;&nbsp;&nbsp; Attention exploration (22 points)</h1>
Multi-headed self-attention is the core modeling component of Transformers. In this question, we’ll get some practice working with the self-attention equations, and motivate why multi-headed self-attention can be preferable to single-headed self-attention.

Recall that attention can be viewed as an operation on a <em>query </em><em>q </em>∈ R<em><sup>d</sup></em>, a set of <em>value </em>vectors {<em>v</em><sub>1</sub><em>,…,v<sub>n</sub></em>}<em>,v<sub>i </sub></em>∈ R<em><sup>d</sup></em>, and a set of <em>key </em>vectors {<em>k</em><sub>1</sub><em>,…,k<sub>n</sub></em>}<em>,k<sub>i </sub></em>∈ R<em><sup>d</sup></em>, specified as follows:

<em>n</em>

<em>c </em>= ∑<em>v<sub>i</sub>α<sub>i&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </sub></em>(1)

<em>i</em>=1

(2)

with <em>α<sub>i </sub></em>termed the “attention weights”. Observe that the output <em>c </em>∈ R<em><sup>d </sup></em>is an average over the value vectors weighted with respect to <em>α<sub>i</sub></em>.

<ul>
<li>(4 points) <strong>Copying in attention. </strong>One advantage of attention is that it’s particularly easy to “copy” a value vector to the output <em>c</em>. In this problem, we’ll motivate why this is the case.
<ol>
<li>(1 point) <strong>Explain </strong>why <em>α </em>can be interpreted as a categorical probability distribution.</li>
<li>(2 points) The distribution <em>α </em>is typically relatively “diffuse”; the probability mass is spread out between many different <em>α<sub>i</sub></em>. However, this is not always the case. <strong>Describe </strong>(in one sentence) under what conditions the categorical distribution <em>α </em>puts almost all of its weight on some <em>α<sub>j</sub></em>, where <em>j </em>∈ {1<em>,…,n</em>} (i.e. <em>α<sub>j </sub></em><sup>≫ </sup>∑<em>i</em><sub≯</sub>=<em><sub>j </sub></em><em>α<sub>i</sub></em>). What must be true about the query <em>q </em>and/or the keys {<em>k</em><sub>1</sub><em>,…,k<sub>n</sub></em>}?</li>
</ol>
</li>
</ul>
<ul>
<li>(1 point) Under the conditions you gave in (ii), <strong>describe </strong>what properties the output <em>c </em>might have. iv. (1 point) <strong>Explain </strong>(in two sentences or fewer) what your answer to (ii) and (iii) means intuitively.</li>
</ul>
<ul>
<li>(7 points) <strong>An average of two. </strong>Instead of focusing on just one vector <em>v<sub>j</sub></em>, a Transformer model might want to incorporate information from <em>multiple </em>source vectors. Consider the case where we instead want to incorporate information from <strong>two </strong>vectors <em>v<sub>a </sub></em>and <em>v<sub>b</sub></em>, with corresponding key vectors <em>k<sub>a </sub></em>and <em>k<sub>b</sub></em>.
<ol>
<li>(3 points) How should we combine two <em>d</em>-dimensional vectors <em>v<sub>a</sub>,v<sub>b </sub></em>into one output vector <em>c </em>in a way that preserves information from both vectors? In machine learning, one common way to do so is to take the average: . It might seem hard to extract information about the original vectors <em>v<sub>a </sub></em>and <em>v<sub>b </sub></em>from the resulting <em>c</em>, but under certain conditions one can do so. In this problem, we’ll see why this is the case.</li>
</ol>
</li>
</ul>
Suppose that although we don’t know <em>v<sub>a </sub></em>or <em>v<sub>b</sub></em>, we do know that <em>v<sub>a </sub></em>lies in a subspace <em>A </em>formed by the <em>m </em>basis vectors {<em>a</em><sub>1</sub><em>,a</em><sub>2</sub><em>,…,a<sub>m</sub></em>}, while <em>v<sub>b </sub></em>lies in a subspace <em>B </em>formed by the <em>p </em>basis vectors {<em>b</em><sub>1</sub><em>,b</em><sub>2</sub><em>,…,b<sub>p</sub></em>}<em>. </em>(This means that any <em>v<sub>a </sub></em>can be expressed as a linear combination of its basis vectors, as can <em>v<sub>b</sub></em>. All basis vectors have norm 1 and orthogonal to each other.) Additionally, suppose that the two subspaces are orthogonal; i.e. <em>a</em>⊤<em><sub>j </sub></em><em>b<sub>k </sub></em>= 0 for all <em>j,k</em>.

Using the basis vectors {<em>a</em><sub>1</sub><em>,a</em><sub>2</sub><em>,…,a<sub>m</sub></em>}, construct a matrix <em>M </em>such that for arbitrary vectors <em>v<sub>a </sub></em>∈ <em>A </em>and <em>v<sub>b </sub></em>∈ <em>B</em>, we can use <em>M </em>to extract <em>v<sub>a </sub></em>from the sum vector <em>s </em>= <em>v<sub>a </sub></em>+ <em>v<sub>b</sub></em>. In other words, we want to construct <em>M </em>such that for any <em>v<sub>a</sub>,v<sub>b</sub></em>, <em>Ms </em>= <em>v<sub>a</sub></em>).

<strong>Note: </strong>both <em>M </em>and <em>v<sub>a</sub>,v<sub>b </sub></em>should be expressed as a vector in R<em><sup>d</sup></em>, not in terms of vectors from <em>A </em>and <em>B</em>.

<strong>Hint: </strong>Given that the vectors {<em>a</em><sub>1</sub><em>,a</em><sub>2</sub><em>,…,a<sub>m</sub></em>} are both <em>orthogonal </em>and <em>form a basis </em>for <em>v<sub>a</sub></em>, we know that there exist some <em>c</em><sub>1</sub><em>,c</em><sub>2</sub><em>,…,c<sub>m </sub></em>such that <em>v<sub>a </sub></em>= <em>c</em><sub>1</sub><em>a</em><sub>1 </sub>+ <em>c</em><sub>2</sub><em>a</em><sub>2 </sub>+ ··· + <em>c<sub>m</sub>a<sub>m</sub></em>. Can you create a vector of these weights <em>c</em>? ii. (4 points) As before, let <em>v<sub>a </sub></em>and <em>v<sub>b </sub></em>be two value vectors corresponding to key vectors <em>k<sub>a </sub></em>and <em>k<sub>b</sub></em>, respectively. Assume that (1) all key vectors are orthogonal, so <em>k<sub>i</sub></em>⊤<em>k<sub>j </sub></em>for all <em>i </em>≠ <em>j</em>; and (2) all key vectors have norm 1.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> <strong>Find an expression </strong>for a query vector <em>q </em>such that.

2

<ul>
<li>(5 points) <strong>Drawbacks of single-headed attention: </strong>In the previous part, we saw how it was <em>possible </em>for a single-headed attention to focus equally on two values. The same concept could easily be extended to any subset of values. In this question we’ll see why it’s not a <em>practical </em> Consider a set of key vectors {<em>k</em><sub>1</sub><em>,…,k<sub>n</sub></em>} that are now randomly sampled, <em>k<sub>i </sub></em>∼ N(<em>µ<sub>i</sub>,</em>Σ<em><sub>i</sub></em>), where the means <em>µ<sub>i </sub></em>∈ R<em><sup>d </sup></em>are known to you, but the covariances Σ<em><sub>i </sub></em>are unknown. Further, assume that the means <em>µ<sub>i </sub></em>are all perpendicular; <em>µ</em>⊤<em><sub>i </sub></em><em>µ<sub>j </sub></em>= 0 if <em>i </em>≠ <em>j</em>, and unit norm, ∥<em>µ<sub>i</sub></em>∥ = 1.
<ol>
<li>(2 points) Assume that the covariance matrices are Σ<em><sub>i </sub></em>= <em>αI</em>∀<em>i </em>∈ {1<em>,</em><a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a><em>,…,n</em>}, for vanishingly small <em>α</em>. Design a query <em>q </em>in terms of the <em>µ<sub>i </sub></em>such that as before,, and provide a brief argument as to why it works.</li>
<li>(3 points) Though single-headed attention is resistant to small perturbations in the keys, some types of larger perturbations may pose a bigger issue. Specifically, in some cases, one key vector <em>k<sub>a </sub></em>may be larger or smaller in norm than the others, while still pointing in the same direction as <em>µ<sub>a</sub></em>. As an example, let us consider a covariance for item for vanishingly small <em>α </em>(as shown in figure 1). This causes <em>k<sub>a </sub></em>to point in roughly the same direction as <em>µ<sub>a</sub></em>, but with large variances in magnitude. Further, let Σ<em><sub>i </sub></em>= <em>αI </em>for all <em>i </em>≠ <em>a</em>.</li>
</ol>
</li>
</ul>
Figure 1: The vector <em>µ<sub>a </sub></em>(shown here in 2D as an example), with the range of possible values of <em>k<sub>a </sub></em>shown in red. As mentioned previously, <em>k<sub>a </sub></em>points in roughly the same direction as <em>µ<sub>a</sub></em>, but may have larger or smaller magnitude.

When you sample {<em>k</em><sub>1</sub><em>,…,k<sub>n</sub></em>} multiple times, and use the <em>q </em>vector that you defined in part i., what qualitatively do you expect the vector <em>c </em>will look like for different samples?

<ul>
<li>(3 points) <strong>Benefits of multi-headed attention: </strong>Now we’ll see some of the power of multi-headed attention. We’ll consider a simple version of multi-headed attention which is identical to singleheaded self-attention as we’ve presented it in this homework, except two query vectors (<em>q</em><sub>1 </sub>and <em>q</em><sub>2</sub>) are defined, which leads to a pair of vectors (<em>c</em><sub>1 </sub>and <em>c</em><sub>2</sub>), each the output of single-headed attention given its respective query vector. The final output of the multi-headed attention is their average,</li>
</ul>
. As in question 1(c), consider a set of key vectors {<em>k</em><sub>1</sub><em>,…,k<sub>n</sub></em>} that are randomly sampled, <em>k<sub>i </sub></em>∼ N(<em>µ<sub>i</sub>,</em>Σ<em><sub>i</sub></em>), where the means <em>µ<sub>i </sub></em>are known to you, but the covariances Σ<em><sub>i </sub></em>are unknown. Also as before, assume that the means <em>µ<sub>i </sub></em>are mutually orthogonal; <em>µ</em>⊤<em><sub>i </sub></em><em>µ<sub>j </sub></em>= 0 if <em>i </em>≠ <em>j</em>, and unit norm, ∥<em>µ<sub>i</sub></em>∥ = 1.

<ol>
<li>(1 point) Assume that the covariance matrices are Σ<em><sub>i </sub></em>= <em>αI</em>, for vanishingly small <em>α</em>. Design <em>q</em><sub>1 </sub>and <em>q</em><sub>2 </sub>such that <em>c </em>is approximately equal to .</li>
<li>(2 points) Assume that the covariance matrices are for vanishingly small <em>α</em>, and Σ<em><sub>i </sub></em>= <em>αI </em>for all <em>i </em>≠ <em>a</em>. Take the query vectors <em>q</em><sub>1 </sub>and <em>q</em><sub>2 </sub>that you designed in part i. What, qualitatively, do you expect the output <em>c </em>to look like across different samples of the key vectors? Please briefly explain why. You can ignore cases in which <em>k<sub>a</sub></em>⊤<em>q<sub>i </sub>&lt; </em>0.</li>
</ol>
<h1>2.&nbsp;&nbsp;&nbsp;&nbsp; Pretrained Transformer models and knowledge access (35 points)</h1>
You’ll train a Transformer to perform a task that involves accessing knowledge about the world – knowledge which isn’t provided via the task’s training data (at least if you want to generalize outside the training set). You’ll find that it more or less fails entirely at the task. You’ll then learn how to pretrain that Transformer on Wikipedia text that contains world knowledge, and find that finetuning that Transformer on the same knowledge-intensive task enables the model to access some of the knowledge learned at pretraining time. You’ll find that this enables models to perform considerably above chance on a held out development set.

The code you’re provided with is a fork of Andrej Karpathy’s <a href="https://github.com/karpathy/minGPT">minGPT</a>. It’s nicer than most research code in that it’s relatively simple and transparent. The “GPT” in minGPT refers to the Transformer language model of OpenAI, originally described in <a href="https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf">this paper</a> [1].

As in previous assignments, you will want to develop on your machine locally, then run training on Azure. You can use the same conda environment from previous assignments for local development, and the same process for training on Azure (see the <a href="https://docs.google.com/document/d/1BQOAjhBxWbywkB4rMFH9iinb6YHSjaWw1TOVlGfyYho">CS224n Azure Guide</a> for a refresher). Specifically, you’ll still be running “conda activate py37_pytorch” on the Azure machine. You’ll need around 5 hours for training, so budget your time accordingly!

Your work with this codebase is as follows:

<ul>
<li>(0 points) <strong>Check out the demo.</strong></li>
</ul>
In the mingpt-demo/ folder is a Jupyter notebook that trains and samples from a Transformer language model. Take a look at it (locally on your computer) to get somewhat familiar with how it defines and trains models. Some of the code you’re writing below will be inspired by what you see in this notebook.

Note that you do not have to write any code or submit written answers for this part.

<ul>
<li>(0 points) <strong>Read through </strong>NameDataset<strong>, our dataset for reading name-birthplace pairs.</strong></li>
</ul>
The task we’ll be working on with our pretrained models is attempting to access the birth place of a notable person, as written in their Wikipedia page. We’ll think of this as a particularly simple form of question answering:

<em>Q: Where was [person] born?</em>

<em>A: [place]</em>

From now on, you’ll be working with the src/ folder. <strong>The code in </strong>mingpt-demo/ <strong>won’t be changed or evaluated for this assignment. </strong>In dataset.py, you’ll find the the class NameDataset, which reads a TSV (tab-separated values) file of name/place pairs and produces examples of the above form that we can feed to our Transformer model.

To get a sense of the examples we’ll be working with, if you run the following code, it’ll load your NameDataset on the training set birth_places_train.tsv and print out a few examples.

python src/dataset.py namedata

Note that you do not have to write any code or submit written answers for this part.

(c) (0 points) <strong>Implement finetuning (without pretraining).</strong>

Take a look at run.py. It has some skeleton code specifying flags you’ll eventually need to handle as command line arguments. In particular, you might want to <em>pretrain</em>, <em>finetune</em>, or <em>evaluate </em>a model with this code. For now, we’ll focus on the finetuning function, in the case without pretraining.

Taking inspiration from the training code in the play_char.ipynb file, write code to finetune a Transformer model on the name/birthplace dataset, via examples from the NameDataset class. For now, implement the case without pretraining (i.e. create a model from scratch and train it on the birthplace prediction task from part (b)). You’ll have to modify two sections, marked [part c] in the code: one to initialize the model, and one to finetune it. Note that you only need to initialize the model in the case labeled “vanilla” for now (later in section (g), we will explore a model variant). Use the hyperparameters for the Trainer specified in the run.py code.

Also take a look at the <em>evaluation </em>code which has been implemented for you. It samples predictions from the trained model and calls evaluate_places() to get the total percentage of correct place predictions. You will run this code in part (d) to evaluate your trained models.

This is an intermediate step for later portions, including Part d, which contains commands you can run to check your implementation. No written answer is required for this part.

<ul>
<li>(5 points) <strong>Make predictions (without pretraining).</strong></li>
</ul>
Train your model on wiki.txt, and evaluate on birth_dev.tsv. Specifically, you should now be able to run the following three commands:

<table width="581">
<tbody>
<tr>
<td width="581"><em># Train on the names dataset</em>

python src/run.py finetune vanilla wiki.txt \

–writing_params_path vanilla.model.params \

–finetune_corpus_path birth_places_train.tsv

<em># Evaluate on the dev set, writing out predictions</em>

python src/run.py evaluate vanilla wiki.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \

–reading_params_path vanilla.model.params \

–eval_corpus_path birth_dev.tsv \

–outputs_path vanilla.nopretrain.dev.predictions

<em># Evaluate on the test set, writing out predictions</em>

python src/run.py evaluate vanilla wiki.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \

–reading_params_path vanilla.model.params \

–eval_corpus_path birth_test_inputs.tsv \

–outputs_path vanilla.nopretrain.<strong>test</strong>.predictions
</td>
</tr>
</tbody>
</table>
Training will take less than 10 minutes (on Azure). Report your model’s accuracy on the dev set (as printed by the second command above). Don’t be surprised if it is well below 10%; we will be digging into why in Part 3. As a reference point, we want to also calculate the accuracy the model would have achieved if it had just predicted “London” as the birth place for everyone in the dev set. Fill in london_baseline.py to calculate the accuracy of that approach and report your result in your write-up. You should be able to leverage existing code such that the file is only a few lines long.

<ul>
<li>(10 points) <strong>Define a <em>span corruption </em>function for pretraining.</strong></li>
</ul>
In the file src/dataset.py, implement the __getitem__() function for the dataset class CharCorruptionDataset. Follow the instructions provided in the comments in dataset.py. Span corruption is explored in the <a href="https://arxiv.org/pdf/1910.10683.pdf">T5 paper</a> [2]. It randomly selects spans of text in a document and replaces them with unique tokens (noising). Models take this noised text, and are required to output a pattern of each unique sentinel followed by the tokens that were replaced by that sentinel in the input. In this question, you’ll implement a simplification that only masks out a single sequence of characters.

This question will be graded via autograder based on whether your span corruption function implements some basic properties of our spec. We’ll instantiate the CharCorruptionDataset with our own data, and draw examples from it.

To help you debug, if you run the following code, it’ll sample a few examples from your CharCorruptionDataset on the pretraining dataset wiki.txt and print them out for you.

python src/dataset.py charcorruption

No written answer is required for this part.

<ul>
<li>(10 points) <strong>Pretrain, finetune, and make predictions. Budget 2 hours for training.</strong></li>
</ul>
Now fill in the <em>pretrain </em>portion of run.py, which will pretrain a model on the span corruption task. Additionally, modify your <em>finetune </em>portion to handle finetuning in the case <em>with </em>pretraining. In particular, if a path to a pretrained model is provided in the bash command, load this model before finetuning it on the birthplace prediction task. Pretrain your model on wiki.txt (which should take approximately two hours), finetune it on NameDataset and evaluate it. Specifically, you should be able to run the following four commands: (Don’t be concerned if the loss appears to plateau in the middle of pretraining; it will eventually go back down.)

<table width="581">
<tbody>
<tr>
<td width="581"><em># Pretrain the model</em>

python src/run.py pretrain vanilla wiki.txt \

–writing_params_path vanilla.pretrain.params

<em># Finetune the model</em>

python src/run.py finetune vanilla wiki.txt \

–reading_params_path vanilla.pretrain.params \

–writing_params_path vanilla.finetune.params \

–finetune_corpus_path birth_places_train.tsv

<em># Evaluate on the dev set; write to disk</em>

python src/run.py evaluate vanilla wiki.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \

–reading_params_path vanilla.finetune.params \

–eval_corpus_path birth_dev.tsv \

–outputs_path vanilla.pretrain.dev.predictions

<em># Evaluate on the test set; write to disk</em>

python src/run.py evaluate vanilla wiki.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \

–reading_params_path vanilla.finetune.params \

–eval_corpus_path birth_test_inputs.tsv \

–outputs_path vanilla.pretrain.<strong>test</strong>.predictions
</td>
</tr>
</tbody>
</table>
Report the accuracy on the dev set (printed by the third command above). We expect the dev accuracy will be at least 10%, and will expect a similar accuracy on the held out test set.

<ul>
<li>(10 points) <strong>Research! Write and try out the <em>synthesizer </em>variant (Budget 2 hours for pretraining!)</strong></li>
</ul>
We’ll now go to changing the Transformer architecture itself – specifically, the self-attention module. While we’ve been using a self-attention scoring function based on dot products, this involves a rather intensive computation that’s quadratic in the sequence length. This is because the dot product between <em>ℓ</em><sup>2 </sup>pairs of word vectors is computed in each computation. <em>Synthesized attention </em>[3] is a very recent alternative that has potential benefits by removing this dot product (and quadratic computation) entirely. It’s a promising idea, and one way for us to ask, “What’s important/right about the Transformer architecture, and where can we improve/prune aspects of it?” In attention.py, implement the forward() method of SynthesizerAttention, which implements a variant of the Synthesizer proposed in the cited paper.

The provided CausalSelfAttention implements the following attention for each head of the multiheaded attention: Let <em>X </em>∈ R<em><sup>ℓ</sup></em>×<em><sup>d </sup></em>(where <em>ℓ </em>is the block size and <em>d </em>is the total dimensionality, <em>d/h </em>is the dimensionality per head.).<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> Let <em>Q,K,V </em>∈ R<em><sup>d</sup></em>×<em><sup>d/h</sup></em>. Then the output of the self-attention head is

<em>Y<sub>i </sub></em>= softmax&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (3)

where <em>Y<sub>i </sub></em>∈ R<em><sup>ℓ</sup></em>×<em><sup>d/h</sup></em>. Then the output of the self-attention is a linear transformation of the concatenation of the heads:

<em>Y </em>= [<em>Y</em><sub>1</sub>;<em>…</em>;<em>Y<sub>h</sub></em>]<em>A&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </em>(4)

where <em>A </em>∈ R<em><sup>d</sup></em>×<em><sup>d </sup></em>and [<em>Y</em><sub>1</sub>;<em>…</em>;<em>Y<sub>h</sub></em>] ∈ R<em><sup>ℓ</sup></em>×<em><sup>d</sup></em>. The code also includes dropout layers which we haven’t written here. We suggest looking at the provided code and noting how this equation is implemented in PyTorch.

Your job is to implement the following variant of attention. Instead of Equation 3, implement the following in SynthesizerAttention:

<em>Y<sub>i </sub></em>= softmax(ReLU(<em>XA<sub>i </sub></em>+ <em>b</em><sub>1</sub>)<em>B<sub>i </sub></em>+ <em>b</em><sub>2</sub>)(<em>XV<sub>i</sub></em>)<em>,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </em>(5)

where <em>A<sub>i </sub></em>∈ R<em><sup>d</sup></em>×<em><sup>d/h</sup></em>, <em>B<sub>i </sub></em>∈ R<em><sup>d/h</sup></em>×<em><sup>ℓ</sup></em>, and <em>V<sub>i </sub></em>∈ R<em><sup>d</sup></em>×<em><sup>d/h</sup></em>.<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a> One way to interpret this is as follows: The term (<em>XQ<sub>i</sub></em>)(<em>XK<sub>i</sub></em>)⊤ is an <em>ℓ </em>× <em>ℓ </em>matrix of attention scores, computed as all pairs of dot products between word embeddings. The synthesizer variant eschews the all-pairs dot product and directly computes the <em>ℓ </em>× <em>ℓ </em>matrix of attention scores by mapping each <em>d</em>-dimensional vector of each head for <em>X </em>to an <em>ℓ</em>-dimesional vector of unnormalized attention weights.

In the rest of the code in the src/ folder, modify your model to support using either CausalSelfAttention or SynthesizerAttention. Add the ability to switch between these attention variants depending on whether “vanilla” (for causal self-attention) or “synthesizer” (for the synthesizer variant) is selected in the command line arguments (see the section marked [part g] in src/run.py). You are free to implement this functionality in any way you choose, so long as it supports these command line arguments.

Below are bash commands that your code should support in order to pretrain the model, finetune it, and make predictions on the dev and test sets. Note that the pretraining process will take approximately 2 hours.

<table width="581">
<tbody>
<tr>
<td width="581"><em># Pretrain the model</em>

python src/run.py pretrain synthesizer wiki.txt \

–writing_params_path synthesizer.pretrain.params
</td>
</tr>
<tr>
<td width="581"><em># Finetune the model</em>

python src/run.py finetune synthesizer wiki.txt \

–reading_params_path synthesizer.pretrain.params \

–writing_params_path synthesizer.finetune.params \

–finetune_corpus_path birth_places_train.tsv

<em># Evaluate on the dev set; write to disk</em>

python src/run.py evaluate synthesizer wiki.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \

–reading_params_path synthesizer.finetune.params \

–eval_corpus_path birth_dev.tsv \

–outputs_path synthesizer.pretrain.dev.predictions

<em># Evaluate on the test set; write to disk</em>

python src/run.py evaluate synthesizer wiki.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \

–reading_params_path synthesizer.finetune.params \

–eval_corpus_path birth_test_inputs.tsv \

–outputs_path synthesizer.pretrain.<strong>test</strong>.predictions
</td>
</tr>
</tbody>
</table>
Report the accuracy of your synthesizer attention model on birthplace prediction on birth_dev.tsv after pretraining and fine-tuning.

<ol>
<li>(8 points) We’ll score your model as to whether it gets at least 5% accuracy on the test set, which has answers held out.</li>
<li>(2 points) Why might the <em>synthesizer </em>self-attention not be able to do, in a single layer, what the key-query-value self-attention can do?</li>
</ol>
<h1>3.&nbsp;&nbsp;&nbsp;&nbsp; Considerations in pretrained knowledge (5 points)</h1>
<strong>Please type the answers to these written questions (to make TA lives easier).</strong>

<ul>
<li>(1 point) Succinctly explain why the pretrained (vanilla) model was able to achieve an accuracy of above 10%, whereas the non-pretrained model was not.</li>
<li>(2 points) Take a look at some of the correct predictions of the pretrain+finetuned vanilla model, as well as some of the errors. We think you’ll find that it’s impossible to tell, just looking at the output, whether the model <em>retrieved </em>the correct birth place, or <em>made up </em>an incorrect birth place. Consider the implications of this for user-facing systems that involve pretrained NLP components. Come up with two <strong>distinct </strong>reasons why this model behavior (i.e. unable to tell whether it’s retrieved or made up) may cause concern for such applications, and an example for each reason.</li>
<li>(2 points) If your model didn’t see a person’s name at pretraining time, and that person was not seen at fine-tuning time either, it is not possible for it to have “learned” where they lived. Yet, your model will produce <em>something </em>as a predicted birth place for that person’s name if asked. Concisely describe a strategy your model might take for predicting a birth place for that person’s name, and one reason why this should cause concern for the use of such applications. (You do not need to submit the same answer for 3c as for 3b.)</li>
</ul>
<h1>Submission Instructions</h1>
You will submit this assignment on GradeScope as two submissions – one for <strong>Assignment 5 [coding] </strong>and another for <strong>Assignment 5 [written]</strong>:

<ol>
<li>Verify that the following files exist at these specified paths within your assignment directory:
<ul>
<li>The no-pretraining model and predictions: vanilla.model.params, vanilla.nopretrain.dev.predictions, vanilla.nopretrain.test.predictions</li>
<li>The pretrain-finetune model and predictions: vanilla.finetune.params, vanilla.pretrain.dev.predictions, vanilla.pretrain.test.predictions</li>
<li>The synthesizer model and predictions: synthesizer.finetune.params, synthesizer.pretrain.dev.predictions, synthesizer.pretrain.test.predictions</li>
</ul>
</li>
<li>Run the collect_submission.sh script to produce your assignment5.zip file.</li>
<li>Upload your assignment5.zip file to GradeScope to <strong>Assignment 5 [coding]</strong>.</li>
<li>Check that the public autograder tests passed correctly.</li>
<li>Upload your written solutions, for questions 1, parts of 2, and 3, to GradeScope to <strong>Assignment 5 [written]</strong>. Tag it properly!</li>
</ol>
<h1>References</h1>
<ul>
<li>Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding with unsupervised learning. <em>Technical report, OpenAI </em>(2018).</li>
<li>Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. <em>Journal of Machine Learning Research 21</em>, 140 (2020), 1–67.</li>
<li>Tay, Y., Bahri, D., Metzler, D., Juan, D.-C., Zhao, Z., and Zheng, C. Synthesizer: Rethinking self-attention in transformer models. <em>arXiv preprint arXiv:2005.00743 </em>(2020).</li>
</ul>
<a href="#_ftnref1" name="_ftn1">[1]</a> Recall that a vector <em>x </em>has norm 1 iff <em>x</em>⊤<em>x</em>=1.

<a href="#_ftnref2" name="_ftn2">[2]</a> Hint: while the softmax function will never <em>exactly </em>average the two vectors, you can get close by using a large scalar multiple in the expression.

<a href="#_ftnref3" name="_ftn3">[3]</a> Note that these dimensionalities do not include the minibatch dimension.

<a href="#_ftnref4" name="_ftn4">[4]</a> Hint: copy over the CausalSelfAttention class, and modify it minimally for this.
