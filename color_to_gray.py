# -*- coding:utf-8 -*-

import numpy as np

func1 = lambda x:x[:, :, :, 0] == 0
func2 = lambda x:x[:, :, :, 1] == 0
func3 = lambda x:x[:, :, :, 2] == 0

func4 = lambda x:x[:, :, :, 0] == 0
func5 = lambda x:x[:, :, :, 1] == 0
func6 = lambda x:x[:, :, :, 2] == 255

func7 = lambda x:x[:, :, :, 0] == 0
func8 = lambda x:x[:, :, :, 1] == 255
func9 = lambda x:x[:, :, :, 2] == 0

func10 = lambda x:x[:, :, :, 0] == 0
func11 = lambda x:x[:, :, :, 1] == 255
func12 = lambda x:x[:, :, :, 2] == 255

func13 = lambda x:x[:, :, :, 0] == 255
func14 = lambda x:x[:, :, :, 1] == 0
func15 = lambda x:x[:, :, :, 2] == 0

func16 = lambda x:x[:, :, :, 0] == 255
func17 = lambda x:x[:, :, :, 1] == 0
func18 = lambda x:x[:, :, :, 2] == 255

func19 = lambda x:x[:, :, :, 0] == 255
func20 = lambda x:x[:, :, :, 1] == 255
func21 = lambda x:x[:, :, :, 2] == 0

func22 = lambda x:x[:, :, :, 0] == 255
func23 = lambda x:x[:, :, :, 1] == 255
func24 = lambda x:x[:, :, :, 2] == 255
