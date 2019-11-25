import numpy as np
import logging

ROW = 19
COL = 19
INIT_BOARD = np.zeros(ROW * COL, dtype=np.int)
INIT_CURRENT_PLAYER = -1
INIT_BOARD[180] = -INIT_CURRENT_PLAYER	# 첫 수로 정중앙에 한 수를 놓음
WIN_COUNT = 6

class Game:

	def __init__(self):
		self.currentPlayer = INIT_CURRENT_PLAYER
		self.gameState = GameState(INIT_BOARD, INIT_CURRENT_PLAYER)
		self.actionSpace = INIT_BOARD
		self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.grid_shape = (ROW, COL)
		self.input_shape = (2, ROW, COL)
		self.name = 'connect6'
		self.state_size = len(self.gameState.binary)
		self.action_size = len(self.actionSpace)

	def reset(self):
		self.gameState = GameState(INIT_BOARD, INIT_CURRENT_PLAYER)
		self.currentPlayer = INIT_CURRENT_PLAYER
		return self.gameState

	def step(self, action):
		next_state, value, done = self.gameState.takeAction(action)
		self.gameState = next_state
		self.currentPlayer = next_state.playerTurn
		info = None
		return ((next_state, value, done, info))

	def identities(self, state, actionValues):
		identities = [(state,actionValues)]

		currentBoard = state.board
		currentAV = actionValues

		identities.append((GameState(currentBoard, state.playerTurn), currentAV))

		return identities


class GameState():
	def __init__(self, board, playerTurn):
		self.board = board
		self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.winners = [
			[0,1,2,3,4,5],
			[1,2,3,4,5,6],
			[2,3,4,5,6,7],
			[3,4,5,6,7,8],
			[4,5,6,7,8,9],
			[5,6,7,8,9,10],
			[6,7,8,9,10,11],
			[7,8,9,10,11,12],
			[8,9,10,11,12,13],
			[9,10,11,12,13,14],
			[10,11,12,13,14,15],
			[11,12,13,14,15,16],
			[12,13,14,15,16,17],
			[13,14,15,16,17,18],
			[19,20,21,22,23,24],
			[20,21,22,23,24,25],
			[21,22,23,24,25,26],
			[22,23,24,25,26,27],
			[23,24,25,26,27,28],
			[24,25,26,27,28,29],
			[25,26,27,28,29,30],
			[26,27,28,29,30,31],
			[27,28,29,30,31,32],
			[28,29,30,31,32,33],
			[29,30,31,32,33,34],
			[30,31,32,33,34,35],
			[31,32,33,34,35,36],
			[32,33,34,35,36,37],
			[38,39,40,41,42,43],
			[39,40,41,42,43,44],
			[40,41,42,43,44,45],
			[41,42,43,44,45,46],
			[42,43,44,45,46,47],
			[43,44,45,46,47,48],
			[44,45,46,47,48,49],
			[45,46,47,48,49,50],
			[46,47,48,49,50,51],
			[47,48,49,50,51,52],
			[48,49,50,51,52,53],
			[49,50,51,52,53,54],
			[50,51,52,53,54,55],
			[51,52,53,54,55,56],
			[57,58,59,60,61,62],
			[58,59,60,61,62,63],
			[59,60,61,62,63,64],
			[60,61,62,63,64,65],
			[61,62,63,64,65,66],
			[62,63,64,65,66,67],
			[63,64,65,66,67,68],
			[64,65,66,67,68,69],
			[65,66,67,68,69,70],
			[66,67,68,69,70,71],
			[67,68,69,70,71,72],
			[68,69,70,71,72,73],
			[69,70,71,72,73,74],
			[70,71,72,73,74,75],
			[76,77,78,79,80,81],
			[77,78,79,80,81,82],
			[78,79,80,81,82,83],
			[79,80,81,82,83,84],
			[80,81,82,83,84,85],
			[81,82,83,84,85,86],
			[82,83,84,85,86,87],
			[83,84,85,86,87,88],
			[84,85,86,87,88,89],
			[85,86,87,88,89,90],
			[86,87,88,89,90,91],
			[87,88,89,90,91,92],
			[88,89,90,91,92,93],
			[89,90,91,92,93,94],
			[95,96,97,98,99,100],
			[96,97,98,99,100,101],
			[97,98,99,100,101,102],
			[98,99,100,101,102,103],
			[99,100,101,102,103,104],
			[100,101,102,103,104,105],
			[101,102,103,104,105,106],
			[102,103,104,105,106,107],
			[103,104,105,106,107,108],
			[104,105,106,107,108,109],
			[105,106,107,108,109,110],
			[106,107,108,109,110,111],
			[107,108,109,110,111,112],
			[108,109,110,111,112,113],
			[114,115,116,117,118,119],
			[115,116,117,118,119,120],
			[116,117,118,119,120,121],
			[117,118,119,120,121,122],
			[118,119,120,121,122,123],
			[119,120,121,122,123,124],
			[120,121,122,123,124,125],
			[121,122,123,124,125,126],
			[122,123,124,125,126,127],
			[123,124,125,126,127,128],
			[124,125,126,127,128,129],
			[125,126,127,128,129,130],
			[126,127,128,129,130,131],
			[127,128,129,130,131,132],
			[133,134,135,136,137,138],
			[134,135,136,137,138,139],
			[135,136,137,138,139,140],
			[136,137,138,139,140,141],
			[137,138,139,140,141,142],
			[138,139,140,141,142,143],
			[139,140,141,142,143,144],
			[140,141,142,143,144,145],
			[141,142,143,144,145,146],
			[142,143,144,145,146,147],
			[143,144,145,146,147,148],
			[144,145,146,147,148,149],
			[145,146,147,148,149,150],
			[146,147,148,149,150,151],
			[152,153,154,155,156,157],
			[153,154,155,156,157,158],
			[154,155,156,157,158,159],
			[155,156,157,158,159,160],
			[156,157,158,159,160,161],
			[157,158,159,160,161,162],
			[158,159,160,161,162,163],
			[159,160,161,162,163,164],
			[160,161,162,163,164,165],
			[161,162,163,164,165,166],
			[162,163,164,165,166,167],
			[163,164,165,166,167,168],
			[164,165,166,167,168,169],
			[165,166,167,168,169,170],
			[171,172,173,174,175,176],
			[172,173,174,175,176,177],
			[173,174,175,176,177,178],
			[174,175,176,177,178,179],
			[175,176,177,178,179,180],
			[176,177,178,179,180,181],
			[177,178,179,180,181,182],
			[178,179,180,181,182,183],
			[179,180,181,182,183,184],
			[180,181,182,183,184,185],
			[181,182,183,184,185,186],
			[182,183,184,185,186,187],
			[183,184,185,186,187,188],
			[184,185,186,187,188,189],
			[190,191,192,193,194,195],
			[191,192,193,194,195,196],
			[192,193,194,195,196,197],
			[193,194,195,196,197,198],
			[194,195,196,197,198,199],
			[195,196,197,198,199,200],
			[196,197,198,199,200,201],
			[197,198,199,200,201,202],
			[198,199,200,201,202,203],
			[199,200,201,202,203,204],
			[200,201,202,203,204,205],
			[201,202,203,204,205,206],
			[202,203,204,205,206,207],
			[203,204,205,206,207,208],
			[209,210,211,212,213,214],
			[210,211,212,213,214,215],
			[211,212,213,214,215,216],
			[212,213,214,215,216,217],
			[213,214,215,216,217,218],
			[214,215,216,217,218,219],
			[215,216,217,218,219,220],
			[216,217,218,219,220,221],
			[217,218,219,220,221,222],
			[218,219,220,221,222,223],
			[219,220,221,222,223,224],
			[220,221,222,223,224,225],
			[221,222,223,224,225,226],
			[222,223,224,225,226,227],
			[228,229,230,231,232,233],
			[229,230,231,232,233,234],
			[230,231,232,233,234,235],
			[231,232,233,234,235,236],
			[232,233,234,235,236,237],
			[233,234,235,236,237,238],
			[234,235,236,237,238,239],
			[235,236,237,238,239,240],
			[236,237,238,239,240,241],
			[237,238,239,240,241,242],
			[238,239,240,241,242,243],
			[239,240,241,242,243,244],
			[240,241,242,243,244,245],
			[241,242,243,244,245,246],
			[247,248,249,250,251,252],
			[248,249,250,251,252,253],
			[249,250,251,252,253,254],
			[250,251,252,253,254,255],
			[251,252,253,254,255,256],
			[252,253,254,255,256,257],
			[253,254,255,256,257,258],
			[254,255,256,257,258,259],
			[255,256,257,258,259,260],
			[256,257,258,259,260,261],
			[257,258,259,260,261,262],
			[258,259,260,261,262,263],
			[259,260,261,262,263,264],
			[260,261,262,263,264,265],
			[266,267,268,269,270,271],
			[267,268,269,270,271,272],
			[268,269,270,271,272,273],
			[269,270,271,272,273,274],
			[270,271,272,273,274,275],
			[271,272,273,274,275,276],
			[272,273,274,275,276,277],
			[273,274,275,276,277,278],
			[274,275,276,277,278,279],
			[275,276,277,278,279,280],
			[276,277,278,279,280,281],
			[277,278,279,280,281,282],
			[278,279,280,281,282,283],
			[279,280,281,282,283,284],
			[285,286,287,288,289,290],
			[286,287,288,289,290,291],
			[287,288,289,290,291,292],
			[288,289,290,291,292,293],
			[289,290,291,292,293,294],
			[290,291,292,293,294,295],
			[291,292,293,294,295,296],
			[292,293,294,295,296,297],
			[293,294,295,296,297,298],
			[294,295,296,297,298,299],
			[295,296,297,298,299,300],
			[296,297,298,299,300,301],
			[297,298,299,300,301,302],
			[298,299,300,301,302,303],
			[304,305,306,307,308,309],
			[305,306,307,308,309,310],
			[306,307,308,309,310,311],
			[307,308,309,310,311,312],
			[308,309,310,311,312,313],
			[309,310,311,312,313,314],
			[310,311,312,313,314,315],
			[311,312,313,314,315,316],
			[312,313,314,315,316,317],
			[313,314,315,316,317,318],
			[314,315,316,317,318,319],
			[315,316,317,318,319,320],
			[316,317,318,319,320,321],
			[317,318,319,320,321,322],
			[323,324,325,326,327,328],
			[324,325,326,327,328,329],
			[325,326,327,328,329,330],
			[326,327,328,329,330,331],
			[327,328,329,330,331,332],
			[328,329,330,331,332,333],
			[329,330,331,332,333,334],
			[330,331,332,333,334,335],
			[331,332,333,334,335,336],
			[332,333,334,335,336,337],
			[333,334,335,336,337,338],
			[334,335,336,337,338,339],
			[335,336,337,338,339,340],
			[336,337,338,339,340,341],
			[342,343,344,345,346,347],
			[343,344,345,346,347,348],
			[344,345,346,347,348,349],
			[345,346,347,348,349,350],
			[346,347,348,349,350,351],
			[347,348,349,350,351,352],
			[348,349,350,351,352,353],
			[349,350,351,352,353,354],
			[350,351,352,353,354,355],
			[351,352,353,354,355,356],
			[352,353,354,355,356,357],
			[353,354,355,356,357,358],
			[354,355,356,357,358,359],
			[355,356,357,358,359,360],

			[0,19,38,57,76,95],
			[19,38,57,76,95,114],
			[38,57,76,95,114,133],
			[57,76,95,114,133,152],
			[76,95,114,133,152,171],
			[95,114,133,152,171,190],
			[114,133,152,171,190,209],
			[133,152,171,190,209,228],
			[152,171,190,209,228,247],
			[171,190,209,228,247,266],
			[190,209,228,247,266,285],
			[209,228,247,266,285,304],
			[228,247,266,285,304,323],
			[247,266,285,304,323,342],
			[1,20,39,58,77,96],
			[20,39,58,77,96,115],
			[39,58,77,96,115,134],
			[58,77,96,115,134,153],
			[77,96,115,134,153,172],
			[96,115,134,153,172,191],
			[115,134,153,172,191,210],
			[134,153,172,191,210,229],
			[153,172,191,210,229,248],
			[172,191,210,229,248,267],
			[191,210,229,248,267,286],
			[210,229,248,267,286,305],
			[229,248,267,286,305,324],
			[248,267,286,305,324,343],
			[2,21,40,59,78,97],
			[21,40,59,78,97,116],
			[40,59,78,97,116,135],
			[59,78,97,116,135,154],
			[78,97,116,135,154,173],
			[97,116,135,154,173,192],
			[116,135,154,173,192,211],
			[135,154,173,192,211,230],
			[154,173,192,211,230,249],
			[173,192,211,230,249,268],
			[192,211,230,249,268,287],
			[211,230,249,268,287,306],
			[230,249,268,287,306,325],
			[249,268,287,306,325,344],
			[3,22,41,60,79,98],
			[22,41,60,79,98,117],
			[41,60,79,98,117,136],
			[60,79,98,117,136,155],
			[79,98,117,136,155,174],
			[98,117,136,155,174,193],
			[117,136,155,174,193,212],
			[136,155,174,193,212,231],
			[155,174,193,212,231,250],
			[174,193,212,231,250,269],
			[193,212,231,250,269,288],
			[212,231,250,269,288,307],
			[231,250,269,288,307,326],
			[250,269,288,307,326,345],
			[4,23,42,61,80,99],
			[23,42,61,80,99,118],
			[42,61,80,99,118,137],
			[61,80,99,118,137,156],
			[80,99,118,137,156,175],
			[99,118,137,156,175,194],
			[118,137,156,175,194,213],
			[137,156,175,194,213,232],
			[156,175,194,213,232,251],
			[175,194,213,232,251,270],
			[194,213,232,251,270,289],
			[213,232,251,270,289,308],
			[232,251,270,289,308,327],
			[251,270,289,308,327,346],
			[5,24,43,62,81,100],
			[24,43,62,81,100,119],
			[43,62,81,100,119,138],
			[62,81,100,119,138,157],
			[81,100,119,138,157,176],
			[100,119,138,157,176,195],
			[119,138,157,176,195,214],
			[138,157,176,195,214,233],
			[157,176,195,214,233,252],
			[176,195,214,233,252,271],
			[195,214,233,252,271,290],
			[214,233,252,271,290,309],
			[233,252,271,290,309,328],
			[252,271,290,309,328,347],
			[6,25,44,63,82,101],
			[25,44,63,82,101,120],
			[44,63,82,101,120,139],
			[63,82,101,120,139,158],
			[82,101,120,139,158,177],
			[101,120,139,158,177,196],
			[120,139,158,177,196,215],
			[139,158,177,196,215,234],
			[158,177,196,215,234,253],
			[177,196,215,234,253,272],
			[196,215,234,253,272,291],
			[215,234,253,272,291,310],
			[234,253,272,291,310,329],
			[253,272,291,310,329,348],
			[7,26,45,64,83,102],
			[26,45,64,83,102,121],
			[45,64,83,102,121,140],
			[64,83,102,121,140,159],
			[83,102,121,140,159,178],
			[102,121,140,159,178,197],
			[121,140,159,178,197,216],
			[140,159,178,197,216,235],
			[159,178,197,216,235,254],
			[178,197,216,235,254,273],
			[197,216,235,254,273,292],
			[216,235,254,273,292,311],
			[235,254,273,292,311,330],
			[254,273,292,311,330,349],
			[8,27,46,65,84,103],
			[27,46,65,84,103,122],
			[46,65,84,103,122,141],
			[65,84,103,122,141,160],
			[84,103,122,141,160,179],
			[103,122,141,160,179,198],
			[122,141,160,179,198,217],
			[141,160,179,198,217,236],
			[160,179,198,217,236,255],
			[179,198,217,236,255,274],
			[198,217,236,255,274,293],
			[217,236,255,274,293,312],
			[236,255,274,293,312,331],
			[255,274,293,312,331,350],
			[9,28,47,66,85,104],
			[28,47,66,85,104,123],
			[47,66,85,104,123,142],
			[66,85,104,123,142,161],
			[85,104,123,142,161,180],
			[104,123,142,161,180,199],
			[123,142,161,180,199,218],
			[142,161,180,199,218,237],
			[161,180,199,218,237,256],
			[180,199,218,237,256,275],
			[199,218,237,256,275,294],
			[218,237,256,275,294,313],
			[237,256,275,294,313,332],
			[256,275,294,313,332,351],
			[10,29,48,67,86,105],
			[29,48,67,86,105,124],
			[48,67,86,105,124,143],
			[67,86,105,124,143,162],
			[86,105,124,143,162,181],
			[105,124,143,162,181,200],
			[124,143,162,181,200,219],
			[143,162,181,200,219,238],
			[162,181,200,219,238,257],
			[181,200,219,238,257,276],
			[200,219,238,257,276,295],
			[219,238,257,276,295,314],
			[238,257,276,295,314,333],
			[257,276,295,314,333,352],
			[11,30,49,68,87,106],
			[30,49,68,87,106,125],
			[49,68,87,106,125,144],
			[68,87,106,125,144,163],
			[87,106,125,144,163,182],
			[106,125,144,163,182,201],
			[125,144,163,182,201,220],
			[144,163,182,201,220,239],
			[163,182,201,220,239,258],
			[182,201,220,239,258,277],
			[201,220,239,258,277,296],
			[220,239,258,277,296,315],
			[239,258,277,296,315,334],
			[258,277,296,315,334,353],
			[12,31,50,69,88,107],
			[31,50,69,88,107,126],
			[50,69,88,107,126,145],
			[69,88,107,126,145,164],
			[88,107,126,145,164,183],
			[107,126,145,164,183,202],
			[126,145,164,183,202,221],
			[145,164,183,202,221,240],
			[164,183,202,221,240,259],
			[183,202,221,240,259,278],
			[202,221,240,259,278,297],
			[221,240,259,278,297,316],
			[240,259,278,297,316,335],
			[259,278,297,316,335,354],
			[13,32,51,70,89,108],
			[32,51,70,89,108,127],
			[51,70,89,108,127,146],
			[70,89,108,127,146,165],
			[89,108,127,146,165,184],
			[108,127,146,165,184,203],
			[127,146,165,184,203,222],
			[146,165,184,203,222,241],
			[165,184,203,222,241,260],
			[184,203,222,241,260,279],
			[203,222,241,260,279,298],
			[222,241,260,279,298,317],
			[241,260,279,298,317,336],
			[260,279,298,317,336,355],
			[14,33,52,71,90,109],
			[33,52,71,90,109,128],
			[52,71,90,109,128,147],
			[71,90,109,128,147,166],
			[90,109,128,147,166,185],
			[109,128,147,166,185,204],
			[128,147,166,185,204,223],
			[147,166,185,204,223,242],
			[166,185,204,223,242,261],
			[185,204,223,242,261,280],
			[204,223,242,261,280,299],
			[223,242,261,280,299,318],
			[242,261,280,299,318,337],
			[261,280,299,318,337,356],
			[15,34,53,72,91,110],
			[34,53,72,91,110,129],
			[53,72,91,110,129,148],
			[72,91,110,129,148,167],
			[91,110,129,148,167,186],
			[110,129,148,167,186,205],
			[129,148,167,186,205,224],
			[148,167,186,205,224,243],
			[167,186,205,224,243,262],
			[186,205,224,243,262,281],
			[205,224,243,262,281,300],
			[224,243,262,281,300,319],
			[243,262,281,300,319,338],
			[262,281,300,319,338,357],
			[16,35,54,73,92,111],
			[35,54,73,92,111,130],
			[54,73,92,111,130,149],
			[73,92,111,130,149,168],
			[92,111,130,149,168,187],
			[111,130,149,168,187,206],
			[130,149,168,187,206,225],
			[149,168,187,206,225,244],
			[168,187,206,225,244,263],
			[187,206,225,244,263,282],
			[206,225,244,263,282,301],
			[225,244,263,282,301,320],
			[244,263,282,301,320,339],
			[263,282,301,320,339,358],
			[17,36,55,74,93,112],
			[36,55,74,93,112,131],
			[55,74,93,112,131,150],
			[74,93,112,131,150,169],
			[93,112,131,150,169,188],
			[112,131,150,169,188,207],
			[131,150,169,188,207,226],
			[150,169,188,207,226,245],
			[169,188,207,226,245,264],
			[188,207,226,245,264,283],
			[207,226,245,264,283,302],
			[226,245,264,283,302,321],
			[245,264,283,302,321,340],
			[264,283,302,321,340,359],
			[18,37,56,75,94,113],
			[37,56,75,94,113,132],
			[56,75,94,113,132,151],
			[75,94,113,132,151,170],
			[94,113,132,151,170,189],
			[113,132,151,170,189,208],
			[132,151,170,189,208,227],
			[151,170,189,208,227,246],
			[170,189,208,227,246,265],
			[189,208,227,246,265,284],
			[208,227,246,265,284,303],
			[227,246,265,284,303,322],
			[246,265,284,303,322,341],
			[265,284,303,322,341,360],

			[0,20,40,60,80,100],
			[1,21,41,61,81,101],
			[2,22,42,62,82,102],
			[3,23,43,63,83,103],
			[4,24,44,64,84,104],
			[5,25,45,65,85,105],
			[6,26,46,66,86,106],
			[7,27,47,67,87,107],
			[8,28,48,68,88,108],
			[9,29,49,69,89,109],
			[10,30,50,70,90,110],
			[11,31,51,71,91,111],
			[12,32,52,72,92,112],
			[13,33,53,73,93,113],
			[19,39,59,79,99,119],
			[20,40,60,80,100,120],
			[21,41,61,81,101,121],
			[22,42,62,82,102,122],
			[23,43,63,83,103,123],
			[24,44,64,84,104,124],
			[25,45,65,85,105,125],
			[26,46,66,86,106,126],
			[27,47,67,87,107,127],
			[28,48,68,88,108,128],
			[29,49,69,89,109,129],
			[30,50,70,90,110,130],
			[31,51,71,91,111,131],
			[32,52,72,92,112,132],
			[38,58,78,98,118,138],
			[39,59,79,99,119,139],
			[40,60,80,100,120,140],
			[41,61,81,101,121,141],
			[42,62,82,102,122,142],
			[43,63,83,103,123,143],
			[44,64,84,104,124,144],
			[45,65,85,105,125,145],
			[46,66,86,106,126,146],
			[47,67,87,107,127,147],
			[48,68,88,108,128,148],
			[49,69,89,109,129,149],
			[50,70,90,110,130,150],
			[51,71,91,111,131,151],
			[57,77,97,117,137,157],
			[58,78,98,118,138,158],
			[59,79,99,119,139,159],
			[60,80,100,120,140,160],
			[61,81,101,121,141,161],
			[62,82,102,122,142,162],
			[63,83,103,123,143,163],
			[64,84,104,124,144,164],
			[65,85,105,125,145,165],
			[66,86,106,126,146,166],
			[67,87,107,127,147,167],
			[68,88,108,128,148,168],
			[69,89,109,129,149,169],
			[70,90,110,130,150,170],
			[76,96,116,136,156,176],
			[77,97,117,137,157,177],
			[78,98,118,138,158,178],
			[79,99,119,139,159,179],
			[80,100,120,140,160,180],
			[81,101,121,141,161,181],
			[82,102,122,142,162,182],
			[83,103,123,143,163,183],
			[84,104,124,144,164,184],
			[85,105,125,145,165,185],
			[86,106,126,146,166,186],
			[87,107,127,147,167,187],
			[88,108,128,148,168,188],
			[89,109,129,149,169,189],
			[95,115,135,155,175,195],
			[96,116,136,156,176,196],
			[97,117,137,157,177,197],
			[98,118,138,158,178,198],
			[99,119,139,159,179,199],
			[100,120,140,160,180,200],
			[101,121,141,161,181,201],
			[102,122,142,162,182,202],
			[103,123,143,163,183,203],
			[104,124,144,164,184,204],
			[105,125,145,165,185,205],
			[106,126,146,166,186,206],
			[107,127,147,167,187,207],
			[108,128,148,168,188,208],
			[114,134,154,174,194,214],
			[115,135,155,175,195,215],
			[116,136,156,176,196,216],
			[117,137,157,177,197,217],
			[118,138,158,178,198,218],
			[119,139,159,179,199,219],
			[120,140,160,180,200,220],
			[121,141,161,181,201,221],
			[122,142,162,182,202,222],
			[123,143,163,183,203,223],
			[124,144,164,184,204,224],
			[125,145,165,185,205,225],
			[126,146,166,186,206,226],
			[127,147,167,187,207,227],
			[133,153,173,193,213,233],
			[134,154,174,194,214,234],
			[135,155,175,195,215,235],
			[136,156,176,196,216,236],
			[137,157,177,197,217,237],
			[138,158,178,198,218,238],
			[139,159,179,199,219,239],
			[140,160,180,200,220,240],
			[141,161,181,201,221,241],
			[142,162,182,202,222,242],
			[143,163,183,203,223,243],
			[144,164,184,204,224,244],
			[145,165,185,205,225,245],
			[146,166,186,206,226,246],
			[152,172,192,212,232,252],
			[153,173,193,213,233,253],
			[154,174,194,214,234,254],
			[155,175,195,215,235,255],
			[156,176,196,216,236,256],
			[157,177,197,217,237,257],
			[158,178,198,218,238,258],
			[159,179,199,219,239,259],
			[160,180,200,220,240,260],
			[161,181,201,221,241,261],
			[162,182,202,222,242,262],
			[163,183,203,223,243,263],
			[164,184,204,224,244,264],
			[165,185,205,225,245,265],
			[171,191,211,231,251,271],
			[172,192,212,232,252,272],
			[173,193,213,233,253,273],
			[174,194,214,234,254,274],
			[175,195,215,235,255,275],
			[176,196,216,236,256,276],
			[177,197,217,237,257,277],
			[178,198,218,238,258,278],
			[179,199,219,239,259,279],
			[180,200,220,240,260,280],
			[181,201,221,241,261,281],
			[182,202,222,242,262,282],
			[183,203,223,243,263,283],
			[184,204,224,244,264,284],
			[190,210,230,250,270,290],
			[191,211,231,251,271,291],
			[192,212,232,252,272,292],
			[193,213,233,253,273,293],
			[194,214,234,254,274,294],
			[195,215,235,255,275,295],
			[196,216,236,256,276,296],
			[197,217,237,257,277,297],
			[198,218,238,258,278,298],
			[199,219,239,259,279,299],
			[200,220,240,260,280,300],
			[201,221,241,261,281,301],
			[202,222,242,262,282,302],
			[203,223,243,263,283,303],
			[209,229,249,269,289,309],
			[210,230,250,270,290,310],
			[211,231,251,271,291,311],
			[212,232,252,272,292,312],
			[213,233,253,273,293,313],
			[214,234,254,274,294,314],
			[215,235,255,275,295,315],
			[216,236,256,276,296,316],
			[217,237,257,277,297,317],
			[218,238,258,278,298,318],
			[219,239,259,279,299,319],
			[220,240,260,280,300,320],
			[221,241,261,281,301,321],
			[222,242,262,282,302,322],
			[228,248,268,288,308,328],
			[229,249,269,289,309,329],
			[230,250,270,290,310,330],
			[231,251,271,291,311,331],
			[232,252,272,292,312,332],
			[233,253,273,293,313,333],
			[234,254,274,294,314,334],
			[235,255,275,295,315,335],
			[236,256,276,296,316,336],
			[237,257,277,297,317,337],
			[238,258,278,298,318,338],
			[239,259,279,299,319,339],
			[240,260,280,300,320,340],
			[241,261,281,301,321,341],
			[247,267,287,307,327,347],
			[248,268,288,308,328,348],
			[249,269,289,309,329,349],
			[250,270,290,310,330,350],
			[251,271,291,311,331,351],
			[252,272,292,312,332,352],
			[253,273,293,313,333,353],
			[254,274,294,314,334,354],
			[255,275,295,315,335,355],
			[256,276,296,316,336,356],
			[257,277,297,317,337,357],
			[258,278,298,318,338,358],
			[259,279,299,319,339,359],
			[260,280,300,320,340,360],

			[5,23,41,59,77,95],
			[6,24,42,60,78,96],
			[7,25,43,61,79,97],
			[8,26,44,62,80,98],
			[9,27,45,63,81,99],
			[10,28,46,64,82,100],
			[11,29,47,65,83,101],
			[12,30,48,66,84,102],
			[13,31,49,67,85,103],
			[14,32,50,68,86,104],
			[15,33,51,69,87,105],
			[16,34,52,70,88,106],
			[17,35,53,71,89,107],
			[18,36,54,72,90,108],
			[24,42,60,78,96,114],
			[25,43,61,79,97,115],
			[26,44,62,80,98,116],
			[27,45,63,81,99,117],
			[28,46,64,82,100,118],
			[29,47,65,83,101,119],
			[30,48,66,84,102,120],
			[31,49,67,85,103,121],
			[32,50,68,86,104,122],
			[33,51,69,87,105,123],
			[34,52,70,88,106,124],
			[35,53,71,89,107,125],
			[36,54,72,90,108,126],
			[37,55,73,91,109,127],
			[43,61,79,97,115,133],
			[44,62,80,98,116,134],
			[45,63,81,99,117,135],
			[46,64,82,100,118,136],
			[47,65,83,101,119,137],
			[48,66,84,102,120,138],
			[49,67,85,103,121,139],
			[50,68,86,104,122,140],
			[51,69,87,105,123,141],
			[52,70,88,106,124,142],
			[53,71,89,107,125,143],
			[54,72,90,108,126,144],
			[55,73,91,109,127,145],
			[56,74,92,110,128,146],
			[62,80,98,116,134,152],
			[63,81,99,117,135,153],
			[64,82,100,118,136,154],
			[65,83,101,119,137,155],
			[66,84,102,120,138,156],
			[67,85,103,121,139,157],
			[68,86,104,122,140,158],
			[69,87,105,123,141,159],
			[70,88,106,124,142,160],
			[71,89,107,125,143,161],
			[72,90,108,126,144,162],
			[73,91,109,127,145,163],
			[74,92,110,128,146,164],
			[75,93,111,129,147,165],
			[81,99,117,135,153,171],
			[82,100,118,136,154,172],
			[83,101,119,137,155,173],
			[84,102,120,138,156,174],
			[85,103,121,139,157,175],
			[86,104,122,140,158,176],
			[87,105,123,141,159,177],
			[88,106,124,142,160,178],
			[89,107,125,143,161,179],
			[90,108,126,144,162,180],
			[91,109,127,145,163,181],
			[92,110,128,146,164,182],
			[93,111,129,147,165,183],
			[94,112,130,148,166,184],
			[100,118,136,154,172,190],
			[101,119,137,155,173,191],
			[102,120,138,156,174,192],
			[103,121,139,157,175,193],
			[104,122,140,158,176,194],
			[105,123,141,159,177,195],
			[106,124,142,160,178,196],
			[107,125,143,161,179,197],
			[108,126,144,162,180,198],
			[109,127,145,163,181,199],
			[110,128,146,164,182,200],
			[111,129,147,165,183,201],
			[112,130,148,166,184,202],
			[113,131,149,167,185,203],
			[119,137,155,173,191,209],
			[120,138,156,174,192,210],
			[121,139,157,175,193,211],
			[122,140,158,176,194,212],
			[123,141,159,177,195,213],
			[124,142,160,178,196,214],
			[125,143,161,179,197,215],
			[126,144,162,180,198,216],
			[127,145,163,181,199,217],
			[128,146,164,182,200,218],
			[129,147,165,183,201,219],
			[130,148,166,184,202,220],
			[131,149,167,185,203,221],
			[132,150,168,186,204,222],
			[138,156,174,192,210,228],
			[139,157,175,193,211,229],
			[140,158,176,194,212,230],
			[141,159,177,195,213,231],
			[142,160,178,196,214,232],
			[143,161,179,197,215,233],
			[144,162,180,198,216,234],
			[145,163,181,199,217,235],
			[146,164,182,200,218,236],
			[147,165,183,201,219,237],
			[148,166,184,202,220,238],
			[149,167,185,203,221,239],
			[150,168,186,204,222,240],
			[151,169,187,205,223,241],
			[157,175,193,211,229,247],
			[158,176,194,212,230,248],
			[159,177,195,213,231,249],
			[160,178,196,214,232,250],
			[161,179,197,215,233,251],
			[162,180,198,216,234,252],
			[163,181,199,217,235,253],
			[164,182,200,218,236,254],
			[165,183,201,219,237,255],
			[166,184,202,220,238,256],
			[167,185,203,221,239,257],
			[168,186,204,222,240,258],
			[169,187,205,223,241,259],
			[170,188,206,224,242,260],
			[176,194,212,230,248,266],
			[177,195,213,231,249,267],
			[178,196,214,232,250,268],
			[179,197,215,233,251,269],
			[180,198,216,234,252,270],
			[181,199,217,235,253,271],
			[182,200,218,236,254,272],
			[183,201,219,237,255,273],
			[184,202,220,238,256,274],
			[185,203,221,239,257,275],
			[186,204,222,240,258,276],
			[187,205,223,241,259,277],
			[188,206,224,242,260,278],
			[189,207,225,243,261,279],
			[195,213,231,249,267,285],
			[196,214,232,250,268,286],
			[197,215,233,251,269,287],
			[198,216,234,252,270,288],
			[199,217,235,253,271,289],
			[200,218,236,254,272,290],
			[201,219,237,255,273,291],
			[202,220,238,256,274,292],
			[203,221,239,257,275,293],
			[204,222,240,258,276,294],
			[205,223,241,259,277,295],
			[206,224,242,260,278,296],
			[207,225,243,261,279,297],
			[208,226,244,262,280,298],
			[214,232,250,268,286,304],
			[215,233,251,269,287,305],
			[216,234,252,270,288,306],
			[217,235,253,271,289,307],
			[218,236,254,272,290,308],
			[219,237,255,273,291,309],
			[220,238,256,274,292,310],
			[221,239,257,275,293,311],
			[222,240,258,276,294,312],
			[223,241,259,277,295,313],
			[224,242,260,278,296,314],
			[225,243,261,279,297,315],
			[226,244,262,280,298,316],
			[227,245,263,281,299,317],
			[233,251,269,287,305,323],
			[234,252,270,288,306,324],
			[235,253,271,289,307,325],
			[236,254,272,290,308,326],
			[237,255,273,291,309,327],
			[238,256,274,292,310,328],
			[239,257,275,293,311,329],
			[240,258,276,294,312,330],
			[241,259,277,295,313,331],
			[242,260,278,296,314,332],
			[243,261,279,297,315,333],
			[244,262,280,298,316,334],
			[245,263,281,299,317,335],
			[246,264,282,300,318,336],
			[252,270,288,306,324,342],
			[253,271,289,307,325,343],
			[254,272,290,308,326,344],
			[255,273,291,309,327,345],
			[256,274,292,310,328,346],
			[257,275,293,311,329,347],
			[258,276,294,312,330,348],
			[259,277,295,313,331,349],
			[260,278,296,314,332,350],
			[261,279,297,315,333,351],
			[262,280,298,316,334,352],
			[263,281,299,317,335,353],
			[264,282,300,318,336,354],
			[265,283,301,319,337,355]
			]
		self.playerTurn = playerTurn
		self.binary = self._binary()
		self.id = self._convertStateToId()
		self.allowedActions = self._allowedActions()
		self.value = self._getValue()
		self.score = self._getScore()

	def _allowedActions(self):
		allowed = []
		for i in range(len(self.board)):
			if self.board[i] == 0:
				allowed.append(i)
		return allowed

	def _binary(self):

		currentplayer_position = np.zeros(len(self.board), dtype=np.int)
		currentplayer_position[self.board==self.playerTurn] = 1

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-self.playerTurn] = 1

		position = np.append(currentplayer_position,other_position)

		return (position)

	def _convertStateToId(self):
		player1_position = np.zeros(len(self.board), dtype=np.int)
		player1_position[self.board==1] = 1

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-1] = 1

		position = np.append(player1_position,other_position)

		id = ''.join(map(str,position))

		return id

	def _get2DToAction(self, row, col):
		return self.board[int(row*COL + col)]

	def _checkForEndGame(self, action):
		stoneNum = np.count_nonzero(self.board)
		if stoneNum == len(self.board):
			return 1
		
		# 돌 12개 이전이면 무조건 return 0
		if stoneNum < 12:
			return 0
		
		recentAction = int(action)
		recentRow = int(recentAction / ROW)
		recentCol = int(recentAction % ROW)
		
		leftBound = recentCol - (WIN_COUNT - 1)
		if leftBound < 0:
			leftBound = 0

		rightBound = recentCol + (WIN_COUNT - 1)
		if rightBound > COL - 1:
			rightBound = COL - 1

		upBound = recentRow - (WIN_COUNT - 1)
		if upBound < 0:
			upBound = 0

		downBound = recentRow + (WIN_COUNT - 1)
		if downBound > ROW - 1:
			downBound = ROW - 1

		# horizon direction	
		for i in range(leftBound, rightBound - (WIN_COUNT - 1) + 1):
			sum = 0
			for j in range(i, i + WIN_COUNT):
				sum += self._get2DToAction(recentRow, j)
			if abs(sum) == 6:
				return 1

		# vertical direction
		for i in range(upBound, downBound - (WIN_COUNT - 1) + 1):
			sum = 0
			for j in range(i, i + WIN_COUNT):
				sum += self._get2DToAction(j, recentCol)
			if abs(sum) == 6:
				return 1

		# \ diagonal direction
		minLeft = leftBound
		minUp = upBound
		maxRight = rightBound
		maxDown = downBound

		if minLeft < minUp:
			minUp = recentRow - (recentCol - minLeft)
		else:
			minLeft = recentCol - (recentRow - minUp)

		if maxRight > maxDown:
			maxDown = recentRow + (maxRight - recentCol)
		else:
			maxRight = recentCol + (maxDown - recentRow)

		for i in range(maxDown - (WIN_COUNT - 1) + 1):
			sum = 0
			for j in range(0, WIN_COUNT):
				sum += self._get2DToAction(minUp + i + j, minLeft + i + j)
			if abs(sum) == 6:
				return 1

		# / diagonal direction
		minRight = rightBound
		minUp = upBound
		maxLeft = leftBound
		maxDown = downBound

		if (COL - 1) - minRight < minUp:
			minUp = recentRow - (minRight - recentCol)
		else:
			minRight = recentCol + (recentRow - minUp)

		if (COL - 1) - maxLeft > maxDown:
			maxDown = recentRow + (recentCol - maxLeft)
		else:
			maxLeft = recentCol + (maxDown - recentRow)

		for i in range(maxDown - (WIN_COUNT - 1) + 1):
			sum = 0
			for j in range(0, WIN_COUNT):
				sum += self._get2DToAction(minUp + i + j, minRight - i - j)
			if abs(sum) == 6:
				return 1
		
		return 0

	def _getValue(self):
		# This is the value of the state for the current player
		# i.e. if the previous player played a winning move, you lose
		for x,y,z,a,b,c in self.winners:
			if (abs(self.board[x] + self.board[y] + self.board[z] + self.board[a] + self.board[b] + self.board[c]) == 6 * -self.playerTurn):
				return (-1, -1, 1)
		return (0, 0, 0)


	def _getScore(self):
		tmp = self.value
		return (tmp[1], tmp[2])




	def takeAction(self, action):
		newBoard = np.array(self.board)
		newBoard[action]=self.playerTurn

		nextPlayerTurn = self.playerTurn * (np.count_nonzero(self.board) % 2 == 0 and -1 or 1)	# 2n 마다 턴 변경 (첫수는 Default로 깔아둠)
		newState = GameState(newBoard, nextPlayerTurn)

		value = 0
		done = 0

		if newState._checkForEndGame(action):
			value = newState.value[0]
			done = 1

		return (newState, value, done) 




	def render(self, logger):
		for r in range(ROW - 1):
			logger.info([self.pieces[str(x)] for x in self.board[COL*r : (COL*r + COL)]])
		logger.info('--------------')
