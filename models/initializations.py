import models

'''
norbf (full), nobases (no gradient), nonsep
uvfa-pos, uvfa-text, cnn+lstm
(nocnn)

rbf + gradient + cnn: full
no rbf: norbf
no gradient: noglobal
no rbf / gradient: nobases
no cnn: nocnn
'''

def init(args, layout_vocab_size, object_vocab_size, text_vocab_size):
    if args.model == 'full': ## new
        model = init_full(args, layout_vocab_size, object_vocab_size, text_vocab_size)
    elif args.model == 'no-gradient':
        model = init_nogradient(args, layout_vocab_size, object_vocab_size, text_vocab_size)
    elif args.model == 'cnn-lstm':
        model = init_cnn_lstm(args, layout_vocab_size, object_vocab_size, text_vocab_size)
    elif args.model == 'uvfa-text':
        model = init_uvfa_text(args, layout_vocab_size, object_vocab_size, text_vocab_size)
    # TODO: clean up UVFA-pos goal loading
    elif args.model == 'uvfa-pos':
        model = init_uvfa_pos(args, layout_vocab_size, object_vocab_size, text_vocab_size)
        train_indices = train_goals
        val_indices = val_goals
    return model


def init_full(args, layout_vocab_size, object_vocab_size, text_vocab_size):
    args.global_coeffs = 3
    args.attention_in_dim = args.obj_embed
    args.lstm_out = args.attention_in_dim * args.attention_out_dim * args.attention_kernel**2 + args.global_coeffs

    state_model = models.LookupModel(layout_vocab_size, args.state_embed).cuda()
    object_model = models.LookupModel(object_vocab_size, args.obj_embed)

    text_model = models.TextModel(text_vocab_size, args.lstm_inp, args.lstm_hid, args.lstm_layers, args.lstm_out)
    heatmap_model = models.AttentionGlobal(text_model, args, map_dim=args.map_dim).cuda()

    model = models.MultiNoRBF(state_model, object_model, heatmap_model, args, map_dim=args.map_dim).cuda()
    return model    


def init_nogradient(args, layout_vocab_size, object_vocab_size, text_vocab_size):
    args.global_coeffs = 0
    args.attention_in_dim = args.obj_embed
    args.lstm_out = args.attention_in_dim * args.attention_out_dim * args.attention_kernel**2
    
    state_model = models.LookupModel(layout_vocab_size, args.state_embed).cuda()
    object_model = models.LookupModel(object_vocab_size, args.obj_embed)

    text_model = models.TextModel(text_vocab_size, args.lstm_inp, args.lstm_hid, args.lstm_layers, args.lstm_out)
    heatmap_model = models.AttentionHeatmap(text_model, args, map_dim=args.map_dim).cuda()

    model = models.MultiNoBases(state_model, object_model, heatmap_model, args, map_dim=args.map_dim).cuda()
    return model


def init_cnn_lstm(args, layout_vocab_size, object_vocab_size, text_vocab_size):
    args.lstm_out = 16
    args.cnn_out_dim = 2*args.lstm_out

    state_model = models.LookupModel(layout_vocab_size, args.state_embed)
    object_model = models.LookupModel(object_vocab_size, args.obj_embed)

    lstm = models.TextModel(text_vocab_size, args.lstm_inp, args.lstm_hid, args.lstm_layers, args.lstm_out)

    model = models.CNN_LSTM(state_model, object_model, lstm, args).cuda()
    return model


def init_uvfa_text(args, layout_vocab_size, object_vocab_size, text_vocab_size, rank = 7):
    print '<Models> Using UVFA variant, consider using a lower learning rate (eg, 0.0001)'
    print '<Models> UVFA rank: {} '.format(rank)

    args.rank = rank
    args.lstm_out = rank

    text_model = models.TextModel(text_vocab_size, args.lstm_inp, args.lstm_hid, args.lstm_layers, args.lstm_out)
    model = models.UVFA_text(text_model, layout_vocab_size, object_vocab_size, args, map_dim=args.map_dim).cuda()
    return model


def init_uvfa_pos(args, layout_vocab_size, object_vocab_size, text_vocab_size, rank = 7):
    print '<Models> Using UVFA variant, consider using a lower learning rate (eg, 0.0001)'
    print '<Models> UVFA rank: {} '.format(rank)

    args.rank = rank
    args.lstm_out = rank

    model = models.UVFA_pos(layout_vocab_size, object_vocab_size, args, map_dim=args.map_dim).cuda()
    return model





