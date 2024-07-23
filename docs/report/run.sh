#!/bin/sh

quarto render report.qmd --to html
open report.html