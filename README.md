
# College Scorecard: Predicting Graduate Earnings

**MATH 456 — Data Science Group Project (Spring 2026)**

## Research Question

What institutional factors best predict graduate earnings, and is the "expensive school premium" real after controlling for selectivity?

## Data Source

[U.S. Department of Education — College Scorecard](https://collegescorecard.ed.gov/data/)

The dataset includes institution-level records on earnings, debt, graduation rates, costs, admissions, and demographics for U.S. colleges and universities.

## Project Structure

```
├── data/               # Raw and processed data (not tracked by git)
├── notebooks/          # Exploratory and analysis notebooks
├── src/                # Reusable code (cleaning, feature engineering, models)
├── reports/            # Final deliverables, figures, and write-ups
└── README.md
```

## Setup

1. Clone the repo
2. Download the College Scorecard data from the link above and place CSVs in `data/raw/`
3. Open the `.Rproj` file in RStudio

## Git Workflow

If you're new to git, here's the standard workflow we'll follow. Run these commands in the **Terminal** tab in RStudio (or any terminal pointed at the project folder).

### First-time setup

```bash
# Clone the repo (only do this once)
git clone <repo-url>
cd MATH456PROJ
```

### Before you start working — always pull the latest changes

```bash
# Switch to main and pull the latest version
git checkout main
git pull
```

This makes sure you're starting from the most up-to-date code. **Always do this before creating a new branch.**

### Create a branch for your work

Never work directly on `main`. Instead, create a branch:

```bash
# Create and switch to a new branch
git checkout -b your-branch-name
```

Use a short, descriptive name like `eda-earnings`, `clean-scorecard-data`, or `model-regression`.

### Make your changes

Edit files, write code, etc. When you're ready to save a checkpoint:

```bash
# See what you've changed
git status

# Stage the files you want to commit
git add filename1.R filename2.R

# Commit with a short message describing what you did
git commit -m "Add exploratory analysis of earnings by institution type"
```

You can make multiple commits on a branch — think of each commit as a save point.

### Push your branch to GitHub

```bash
# First push of a new branch
git push -u origin your-branch-name

# Subsequent pushes on the same branch
git push
```

### Open a Pull Request (PR)

1. Go to the repo on GitHub
2. You'll see a banner saying your branch was recently pushed — click **"Compare & pull request"**
3. Add a short title and description of what you did
4. Click **"Create pull request"**
5. Let the team know so someone can review it

Once approved, merge the PR on GitHub and delete the branch.

### After your PR is merged

```bash
# Switch back to main and pull the merged changes
git checkout main
git pull
```

Now you're ready to start a new branch for your next task.

### Quick reference

| What you want to do              | Command                                  |
|----------------------------------|------------------------------------------|
| Check what branch you're on      | `git branch`                             |
| See what files changed           | `git status`                             |
| See the actual changes           | `git diff`                               |
| Switch to an existing branch     | `git checkout branch-name`               |
| Undo changes to a file (careful) | `git checkout -- filename`               |
| View commit history              | `git log --oneline`                      |

## Team


*Processing the data* 

Once the your repo is synced with main and youve downloaded and extracted all the data. 
