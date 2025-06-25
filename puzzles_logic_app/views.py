from django.shortcuts import render
import logging

logger = logging.getLogger(__name__)


def main_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'main_introduction'
    }
    logger.debug("Rendering Main Introduction for Puzzles & Real-World Logic.")
    return render(request, 'puzzles_logic_app/puzzles_logic_introduction.html', context)


def diet_problem_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'diet_problem_introduction'
    }
    logger.debug("Rendering Diet Problem introduction page.")
    return render(request, 'puzzles_logic_app/diet_problem_introduction.html', context)


def diet_problem_demo_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'diet_problem_demo'
    }
    logger.debug("Rendering Diet Problem demo page.")
    return render(request, 'puzzles_logic_app/diet_problem_demo.html', context)

# --- 2. Sports Scheduling ---
def sports_scheduling_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'sports_scheduling_introduction'
    }
    logger.debug("Rendering Sports Scheduling introduction page.")
    return render(request, 'puzzles_logic_app/sports_scheduling_introduction.html', context)

def sports_scheduling_demo_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'sports_scheduling_demo'
    }
    logger.debug("Rendering Sports Scheduling demo page.")
    return render(request, 'puzzles_logic_app/sports_scheduling_demo.html', context)

# --- 3. Nurse Rostering Problem ---
def nurse_rostering_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'nurse_rostering_introduction'
    }
    logger.debug("Rendering Nurse Rostering introduction page.")
    return render(request, 'puzzles_logic_app/nurse_rostering_introduction.html', context)

def nurse_rostering_demo_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'nurse_rostering_demo'
    }
    logger.debug("Rendering Nurse Rostering demo page.")
    return render(request, 'puzzles_logic_app/nurse_rostering_demo.html', context)

# --- 4. Traveling Salesman Problem (TSP) ---
def tsp_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'tsp_introduction'
    }
    logger.debug("Rendering TSP introduction page.")
    return render(request, 'puzzles_logic_app/tsp_introduction.html', context)

def tsp_demo_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'tsp_demo'
    }
    logger.debug("Rendering TSP demo page.")
    return render(request, 'puzzles_logic_app/tsp_demo.html', context)

# --- 5. Sudoku Solver ---
def sudoku_introduction_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'sudoku_introduction'
    }
    logger.debug("Rendering Sudoku introduction page.")
    return render(request, 'puzzles_logic_app/sudoku_introduction.html', context)

def sudoku_demo_view(request):
    context = {
        'active_model': 'Puzzles & Real-World Logic',
        'active_submenu': 'sudoku_demo'
    }
    logger.debug("Rendering Sudoku demo page.")
    return render(request, 'puzzles_logic_app/sudoku_demo.html', context)



