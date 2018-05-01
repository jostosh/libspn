from context import libspn as spn
from test import TestCase
import libspn.generation.dynamic as dspn
import tensorflow as tf
import numpy as np
from parameterized import parameterized, param
import itertools


MAX_STEPS = 8
BATCH_SIZE = 8


def arg_product(*args):
    return [param(*vals) for vals in itertools.product(*args)]


def get_dspn(max_steps=MAX_STEPS, iv_inputs=True):

    tf.set_random_seed(1234)

    weight_init_value = spn.ValueType.RANDOM_UNIFORM()

    if iv_inputs:
        ix = spn.DynamicIVs(name="iv_x", num_vars=1, num_vals=2, max_steps=max_steps)
        iy = spn.DynamicIVs(name="iv_y", num_vars=1, num_vals=2, max_steps=max_steps)
        iz = spn.DynamicIVs(name="iv_z", num_vars=1, num_vals=2, max_steps=max_steps)

        x_in, y_in, z_in = [ix, iy, iz]
        nw = 2
    else:
        ix = spn.DynamicContVars(name="iv_x", num_vars=2, max_steps=max_steps)
        iy = spn.DynamicContVars(name="iv_y", num_vars=2, max_steps=max_steps)
        iz = spn.DynamicContVars(name="iv_z", num_vars=2, max_steps=max_steps)

        x_in = spn.Product(ix, name="ProdX")
        y_in = spn.Product(iy, name="ProdY")
        z_in = spn.Product(iz, name="ProdZ")
        
        nw = 1

    mixture_x0_w = spn.Weights(num_weights=nw, name="mixture_x0_w", init_value=weight_init_value)

    # First define template network
    mix_x0 = spn.Sum(x_in, name="mixture_x0")
    mix_x0.set_weights(mixture_x0_w)

    mixture_x1_w = spn.Weights(num_weights=nw, name="mixture_x1_w", init_value=weight_init_value)
    mix_x1 = spn.Sum(x_in, name="mixture_x1")
    mix_x1.set_weights(mixture_x1_w)

    mixture_y0_w = spn.Weights(num_weights=nw, name="mixture_y0_w", init_value=weight_init_value)
    mix_y0 = spn.Sum(y_in, name="mixture_y0")
    mix_y0.set_weights(mixture_y0_w)

    mixture_y1_w = spn.Weights(num_weights=nw, name="mixture_y1_w", init_value=weight_init_value)
    mix_y1 = spn.Sum(y_in, name="mixture_y1")
    mix_y1.set_weights(mixture_y1_w)

    mixture_z0_w = spn.Weights(num_weights=nw, name="mixture_z0_w", init_value=weight_init_value)
    mix_z0 = spn.Sum(z_in, name="mixture_z0")
    mix_z0.set_weights(mixture_z0_w)

    mixture_z1_w = spn.Weights(num_weights=nw, name="mixture_z1_w", init_value=weight_init_value)
    mix_z1 = spn.Sum(z_in, name="mixture_z1")
    mix_z1.set_weights(mixture_z1_w)

    # Define interface network
    intf0 = spn.DynamicInterface(name="interface0")
    intf1 = spn.DynamicInterface(name="interface1")

    mixture_in0_w = spn.Weights(num_weights=2, name="mixture_in0_w", init_value=weight_init_value)
    mix_int0 = spn.Sum(intf0, intf1, name="mixture_intf0", interface_head=True)
    mix_int0.set_weights(mixture_in0_w)

    mixture_in1_w = spn.Weights(num_weights=2, name="mixture_in1_w", init_value=weight_init_value)
    mix_int1 = spn.Sum(intf0, intf1, name="mixture_intf1", interface_head=True)
    mix_int1.set_weights(mixture_in1_w)

    # Define template heads
    prod0 = spn.Product(mix_x0, mix_y0, mix_z0, name="prod0")
    prod1 = spn.Product(mix_x1, mix_y1, mix_z1, name="prod1")

    # Register sources for interface nodes
    intf0.set_source(prod0)
    intf1.set_source(prod1)

    prod0.add_values(mix_int0)
    prod1.add_values(mix_int1)

    # Define top network
    top_weights = spn.Weights(num_weights=2, name="top_w")
    top_net = spn.Sum(prod0, prod1, name="top")
    top_net.set_weights(top_weights)

    return top_net, [ix, iy, iz], [
        mixture_x0_w, mixture_x1_w,
        mixture_y0_w, mixture_y1_w,
        mixture_z0_w, mixture_z1_w,
        mixture_in0_w, mixture_in1_w,
        top_weights]


def get_dspn_unrolled(max_steps=MAX_STEPS, iv_inputs=True):
    template_network = dspn.TemplateNetwork()

    weight_init_value = spn.ValueType.RANDOM_UNIFORM()

    tf.set_random_seed(1234)
    if iv_inputs:
        ix = in_x = template_network.add_ivs("iv_x", num_vars=1, num_vals=2)
        iy = in_y = template_network.add_ivs("iv_y", num_vars=1, num_vals=2)
        iz = in_z = template_network.add_ivs("iv_z", num_vars=1, num_vals=2)
        
        nw = 2
    else:
        cx = template_network.add_cont_vars("cx", num_vars=2)
        cy = template_network.add_cont_vars("cy", num_vars=2)
        cz = template_network.add_cont_vars("cz", num_vars=2)

        in_x = template_network.add_product("prodx", inputs=cx)
        in_y = template_network.add_product("prody", inputs=cy)
        in_z = template_network.add_product("prodz", inputs=cz)
        
        nw = 1

    mixture_x0_w = spn.Weights(num_weights=nw, name="mixture_x0_w", init_value=weight_init_value)
    mixture_x0 = template_network.add_sum("mix_x0", inputs=in_x, weights=mixture_x0_w)
    mixture_x1_w = spn.Weights(num_weights=nw, name="mixture_x1_w", init_value=weight_init_value)
    mixture_x1 = template_network.add_sum("mix_x1", inputs=in_x, weights=mixture_x1_w)

    mixture_y0_w = spn.Weights(num_weights=nw, name="mixture_y0_w", init_value=weight_init_value)
    mixture_y0 = template_network.add_sum("mix_y0", inputs=in_y, weights=mixture_y0_w)
    mixture_y1_w = spn.Weights(num_weights=nw, name="mixture_y1_w", init_value=weight_init_value)
    mixture_y1 = template_network.add_sum("mix_y1", inputs=in_y, weights=mixture_y1_w)

    mixture_z0_w = spn.Weights(num_weights=nw, name="mixture_z0_w", init_value=weight_init_value)
    mixture_z0 = template_network.add_sum("mix_z0", inputs=in_z, weights=mixture_z0_w)
    mixture_z1_w = spn.Weights(num_weights=nw, name="mixture_z1_w", init_value=weight_init_value)
    mixture_z1 = template_network.add_sum("mix_z1", inputs=in_z, weights=mixture_z1_w)

    interface_forward_declaration = [dspn.TemplateNetwork.InterfaceNode("Interface0", 0),
                                     dspn.TemplateNetwork.InterfaceNode("Interface1", 1)]
    mixture_in0_w = spn.Weights(num_weights=2, name="mixture_in0_w", init_value=weight_init_value)
    mixture_in0 = template_network.add_sum(
        "mix_interface0", interface=interface_forward_declaration, weights=mixture_in0_w)
    mixture_in1_w = spn.Weights(num_weights=2, name="mixture_in1_w", init_value=weight_init_value)
    mixture_in1 = template_network.add_sum(
        "mix_interface1", interface=interface_forward_declaration, weights=mixture_in1_w)

    prod0 = template_network.add_product(
        "prod0", inputs=[mixture_x0, mixture_y0, mixture_z0, mixture_in0], is_root=True)
    prod1 = template_network.add_product(
        "prod1", inputs=[mixture_x1, mixture_y1, mixture_z1, mixture_in1], is_root=True)

    top_network = dspn.TopNetwork()
    top_weights = spn.Weights(num_weights=2, name="top_w")
    top_root = top_network.add_sum("top_root", interface=interface_forward_declaration,
                                   is_root=True, weights=top_weights)

    template_steps = template_network.build_n_steps(max_steps)
    top_steps = top_network.build(template_steps)

    return top_steps[-1].root_nodes[0], [step.var_nodes for step in template_steps], \
        [mixture_x0_w, mixture_x1_w,
         mixture_y0_w, mixture_y1_w,
         mixture_z0_w, mixture_z1_w,
         mixture_in0_w, mixture_in1_w,
         top_weights]


def get_data(max_steps=MAX_STEPS, iv_inputs=True, batch_size=BATCH_SIZE):
    ix_feed = []
    iy_feed = []
    iz_feed = []
    for _ in range(max_steps):
        if iv_inputs:
            ix_feed.append(np.random.randint(-1, 2, size=batch_size).reshape(batch_size, 1))
            iy_feed.append(np.random.randint(-1, 2, size=batch_size).reshape(batch_size, 1))
            iz_feed.append(np.random.randint(-1, 2, size=batch_size).reshape(batch_size, 1))
        else:
            ix_feed.append(np.random.rand(batch_size, 2))
            iy_feed.append(np.random.rand(batch_size, 2))
            iz_feed.append(np.random.rand(batch_size, 2))

    return [ix_feed, iy_feed, iz_feed]


def get_feed_dicts(var_nodes, dynamic_var_nodes, iv_inputs, varlen=False):
    ix_feed, iy_feed, iz_feed = get_data(iv_inputs=iv_inputs)

    unrolled_feed = {}

    if varlen:
        for b, var_nodes_step in enumerate(var_nodes):
            for i, step in enumerate(var_nodes_step):
                t = MAX_STEPS - len(var_nodes_step) + i
                for node, val in zip(step, [ix_feed[t][b:b+1], iy_feed[t][b:b+1], iz_feed[t][b:b+1]]):
                    unrolled_feed[node] = val
    else:
        for i, step in enumerate(var_nodes):
            for node, val in zip(step, [ix_feed[i], iy_feed[i], iz_feed[i]]):
                unrolled_feed[node] = val

    dynamic_feed = {node: np.stack(feed) for node, feed in
                    zip(dynamic_var_nodes, [ix_feed, iy_feed, iz_feed])}
    return unrolled_feed, dynamic_feed


class TestDSPN(TestCase):

    @parameterized.expand(arg_product(
        [False, True], [spn.InferenceType.MPE, spn.InferenceType.MARGINAL], [False, True],
        [False, True]))
    def test_value(self, log, inf_type, iv_inputs, varlen):
        sequence_lens = np.random.randint(1, 1 + MAX_STEPS, size=BATCH_SIZE) if varlen else None
        sequence_lens_ph = tf.placeholder(tf.int32, [None]) if varlen else None

        dynamic_root, dynamic_var_nodes, dynamic_weights = get_dspn(iv_inputs=iv_inputs)
        init_dynamic = spn.initialize_weights(dynamic_root)

        self.assertTrue(dynamic_root.is_valid())

        if varlen:
            unrolled_root_all, var_nodes_all, unrolled_weights_all = [], [], []
            copy_weight_ops, init_weight_ops = [], []
            for len in sequence_lens:
                unrolled_root, var_nodes, unrolled_weights = get_dspn_unrolled(
                    iv_inputs=iv_inputs, max_steps=len)
                unrolled_root_all.append(unrolled_root)
                var_nodes_all.append(var_nodes)
                unrolled_weights_all.append(unrolled_weights)
                copy_weight_ops.extend([tf.assign(uw.variable, dw.variable) for dw, uw in
                                        zip(dynamic_weights, unrolled_weights)])

            copy_weights = tf.group(*copy_weight_ops)
            unrolled_feed, dynamic_feed = get_feed_dicts(
                var_nodes_all, dynamic_var_nodes, iv_inputs, varlen=True)
            dynamic_feed[sequence_lens_ph] = sequence_lens

        else:
            unrolled_root, var_nodes, unrolled_weights = get_dspn_unrolled(iv_inputs=iv_inputs)

            copy_weights = tf.group(*[tf.assign(uw.variable, dw.variable) for dw, uw in
                                      zip(dynamic_weights, unrolled_weights)])

            unrolled_feed, dynamic_feed = get_feed_dicts(var_nodes, dynamic_var_nodes, iv_inputs)

        if not log:
            dval = spn.DynamicValue(inf_type).get_value(
                dynamic_root, sequence_lens=sequence_lens)
            if varlen:
                uval = tf.concat(
                    [spn.Value(inf_type).get_value(root) for root in unrolled_root_all], axis=0)
            else:
                uval = spn.Value(inf_type).get_value(unrolled_root)
        else:
            dval = spn.DynamicLogValue(inf_type).get_value(
                dynamic_root, sequence_lens=sequence_lens)
            if varlen:
                uval = tf.concat(
                    [spn.LogValue(inf_type).get_value(root) for root in unrolled_root_all], axis=0)
            else:
                uval = spn.LogValue(inf_type).get_value(unrolled_root)

        with self.test_session() as sess:
            sess.run([init_dynamic])
            sess.run([copy_weights])

            dynamic_out = sess.run(dval, feed_dict=dynamic_feed)
            unrolled_out = sess.run(uval, feed_dict=unrolled_feed)

        self.assertAllClose(dynamic_out, unrolled_out)

    @parameterized.expand(arg_product(
        [True, False], [spn.InferenceType.MPE, spn.InferenceType.MARGINAL], [False, True],
        [False, True]))
    def test_mpe_path(self, log, inf_type, iv_inputs, varlen):
        sequence_lens = np.random.randint(1, 1 + MAX_STEPS, size=BATCH_SIZE) if varlen else None
        sequence_lens_ph = tf.placeholder(tf.int32, [None]) if varlen else None

        dynamic_root, dynamic_var_nodes, dynamic_weights = get_dspn(iv_inputs=iv_inputs)
        if varlen:
            unrolled_root_all, var_nodes_all, unrolled_weights_all = [], [], []
            copy_weight_ops, init_weight_ops = [], []
            for len in sequence_lens:
                unrolled_root, var_nodes, unrolled_weights = get_dspn_unrolled(
                    iv_inputs=iv_inputs, max_steps=len)
                unrolled_root_all.append(unrolled_root)
                var_nodes_all.append(var_nodes)
                unrolled_weights_all.append(unrolled_weights)
                copy_weight_ops.extend([tf.assign(uw.variable, dw.variable) for dw, uw in
                                        zip(dynamic_weights, unrolled_weights)])

            copy_weights = tf.group(*copy_weight_ops)
            unrolled_feed, dynamic_feed = get_feed_dicts(
                var_nodes_all, dynamic_var_nodes, iv_inputs, varlen=True)
            dynamic_feed[sequence_lens_ph] = sequence_lens

        else:
            unrolled_root, var_nodes, unrolled_weights = get_dspn_unrolled(iv_inputs=iv_inputs)

            copy_weights = tf.group(*[tf.assign(uw.variable, dw.variable) for dw, uw in
                                      zip(dynamic_weights, unrolled_weights)])

            unrolled_feed, dynamic_feed = get_feed_dicts(var_nodes, dynamic_var_nodes, iv_inputs)
        init_dynamic = spn.initialize_weights(dynamic_root)

        pathgen_dynamic = spn.MPEPath(
            log=log, dynamic=True, dynamic_reduce_in_loop=False, value_inference_type=inf_type)
        pathgen_unrolled = spn.MPEPath(log=log, value_inference_type=inf_type)

        pathgen_dynamic.get_mpe_path(dynamic_root, sequence_lens=sequence_lens_ph)
        # print(pathgen_dynamic.counts_per_step.keys())
        counts_per_step_dynamic = [pathgen_dynamic.counts_per_step[n] for n
                                   in dynamic_var_nodes]

        counts_total_dynamic = [pathgen_dynamic.counts[n] for n in dynamic_var_nodes]

        if not varlen:
            pathgen_unrolled.get_mpe_path(unrolled_root)

            xs, ys, zs = [], [], []
            for var_nodes_per_step in var_nodes:
                xs.append(pathgen_unrolled.counts[var_nodes_per_step[0]])
                ys.append(pathgen_unrolled.counts[var_nodes_per_step[1]])
                zs.append(pathgen_unrolled.counts[var_nodes_per_step[2]])
            counts_per_step_unrolled = [tf.stack(l) for l in [xs, ys, zs]]
        else:
            xs_batch, ys_batch, zs_batch = [], [], []
            for unrolled_root, var_nodes in zip(unrolled_root_all, var_nodes_all):
                pathgen_unrolled = spn.MPEPath(log=log, value_inference_type=inf_type)
                pathgen_unrolled.get_mpe_path(unrolled_root)
                xs, ys, zs = [], [], []
                for var_nodes_per_step in var_nodes:
                    xs.append(pathgen_unrolled.counts[var_nodes_per_step[0]])
                    ys.append(pathgen_unrolled.counts[var_nodes_per_step[1]])
                    zs.append(pathgen_unrolled.counts[var_nodes_per_step[2]])
                xs_batch.append(tf.add_n(xs))
                ys_batch.append(tf.add_n(ys))
                zs_batch.append(tf.add_n(zs))
            counts_total_unrolled = [tf.concat(arr, axis=0)
                                     for arr in [xs_batch, ys_batch, zs_batch]]

        with self.test_session() as sess:
            sess.run([init_dynamic])
            sess.run(copy_weights)

            if varlen:
                dynamic_out = sess.run(counts_total_dynamic, feed_dict=dynamic_feed)
                unrolled_out = sess.run(counts_total_unrolled, feed_dict=unrolled_feed)
            else:
                dynamic_out = sess.run(counts_per_step_dynamic, feed_dict=dynamic_feed)
                unrolled_out = sess.run(counts_per_step_unrolled, feed_dict=unrolled_feed)

        for node, do, uo in zip(dynamic_var_nodes, dynamic_out, unrolled_out):
            self.assertAllClose(do, uo)

    @parameterized.expand(arg_product(
        [True, False], [spn.InferenceType.MPE, spn.InferenceType.MARGINAL], [False, True],
        [True, False]))
    def test_mpe_state(self, log, inf_type, iv_inputs, varlen):
        sequence_lens = np.random.randint(1, 1 + MAX_STEPS, size=BATCH_SIZE) if varlen else None
        sequence_lens_ph = tf.placeholder(tf.int32, [None]) if varlen else None

        dynamic_root, dynamic_var_nodes, dynamic_weights = get_dspn(iv_inputs=iv_inputs)

        latent_feed = np.random.randint(-1, 2, size=BATCH_SIZE).reshape((BATCH_SIZE, 1))
        if varlen:
            unrolled_root_all, var_nodes_all, unrolled_weights_all = [], [], []
            copy_weight_ops, init_weight_ops = [], []
            for seq_len in sequence_lens:
                unrolled_root, var_nodes, unrolled_weights = get_dspn_unrolled(
                    iv_inputs=iv_inputs, max_steps=seq_len)
                unrolled_root_all.append(unrolled_root)
                var_nodes_all.append(var_nodes)
                unrolled_weights_all.append(unrolled_weights)
                copy_weight_ops.extend([tf.assign(uw.variable, dw.variable) for dw, uw in
                                        zip(dynamic_weights, unrolled_weights)])

            copy_weights = tf.group(*copy_weight_ops)
            unrolled_feed, dynamic_feed = get_feed_dicts(
                var_nodes_all, dynamic_var_nodes, iv_inputs, varlen=True)
            dynamic_feed[sequence_lens_ph] = sequence_lens

            latent_unr = [root.generate_ivs() for root in unrolled_root_all]
            for node, feed in zip(latent_unr, latent_feed):
                unrolled_feed[node] = np.expand_dims(feed, 0)
        else:
            unrolled_root, var_nodes, unrolled_weights = get_dspn_unrolled(iv_inputs=iv_inputs)

            copy_weights = tf.group(*[tf.assign(uw.variable, dw.variable) for dw, uw in
                                      zip(dynamic_weights, unrolled_weights)])

            unrolled_feed, dynamic_feed = get_feed_dicts(var_nodes, dynamic_var_nodes, iv_inputs)

            latent_unr = unrolled_root.generate_ivs()
            unrolled_feed[latent_unr] = latent_feed

        init_dynamic = spn.initialize_weights(dynamic_root)

        latent_dyn = dynamic_root.generate_ivs()
        dynamic_feed[latent_dyn] = latent_feed

        mpe_state_gen_dynamic = spn.MPEState(log=log, dynamic=True, value_inference_type=inf_type)

        mpe_latent_dyn, *mpe_ivs_dyn = mpe_state_gen_dynamic.get_state(
            dynamic_root, latent_dyn, *dynamic_var_nodes, sequence_lens=sequence_lens_ph)
        if not varlen:
            mpe_state_gen_unrolled = spn.MPEState(
                log=log, dynamic=False, value_inference_type=inf_type)
            mpe_latent_unr, *mpe_ivs_unr = mpe_state_gen_unrolled.get_state(
                unrolled_root, latent_unr, *list(itertools.chain(*var_nodes)))
        else:
            mpe_ivs_unr_varlen = []
            mpe_latent_unr_varlen = []
            for unrolled_root, latent_node, var_nodes in zip(
                    unrolled_root_all, latent_unr, var_nodes_all):
                mpe_state_gen_unrolled = spn.MPEState(
                    log=log, dynamic=False, value_inference_type=inf_type)
                mpe_latent_unr, *mpe_ivs_unr = mpe_state_gen_unrolled.get_state(
                    unrolled_root, latent_node, *list(itertools.chain(*var_nodes)))
                mpe_latent_unr_varlen.append(mpe_latent_unr)
                mpe_ivs_unr_varlen.append(mpe_ivs_unr)

        with self.test_session() as sess:
            sess.run(init_dynamic)
            sess.run(copy_weights)

            unrolled_weights_val = sess.run([w.variable for w in unrolled_weights])
            dynamic_weights_val = sess.run([w.variable for w in dynamic_weights])

            for wu, wd in zip(unrolled_weights_val, dynamic_weights_val):
                self.assertAllClose(wu, wd)

            *mpe_ivs_val_dyn, mpe_latent_val_dyn = sess.run(
                mpe_ivs_dyn + [mpe_latent_dyn], feed_dict=dynamic_feed)
            if not varlen:
                *mpe_ivs_val_unr, mpe_latent_val_unr = sess.run(
                    mpe_ivs_unr + [mpe_latent_unr], feed_dict=unrolled_feed)

                mpe_ivs_val_unr = [np.stack(mpe_ivs_val_unr[i::3]) for i in range(3)]
            else:
                mpe_ivs_val_unr_all, mpe_latent_val_unr_all = [], []
                for seq_len, mpe_latent_unr, mpe_ivs_unr in zip(
                        sequence_lens, mpe_latent_unr_varlen, mpe_ivs_unr_varlen):
                    *mpe_ivs_val_unr, mpe_latent_val_unr = sess.run(
                        mpe_ivs_unr + [mpe_latent_unr], feed_dict=unrolled_feed)

                    mpe_ivs_val_unr = [np.concatenate([
                        np.zeros((MAX_STEPS - seq_len,) + mpe_ivs_val_unr[i].shape),
                        np.stack(mpe_ivs_val_unr[i::3])])
                        for i in range(3)]
                    mpe_ivs_val_unr_all.extend(mpe_ivs_val_unr)
                    mpe_latent_val_unr_all.append(mpe_latent_val_unr)

                mpe_ivs_val_unr = [
                    np.concatenate(mpe_ivs_val_unr_all[i::3], axis=1) for i in range(3)]

                mpe_latent_val_unr = np.concatenate(mpe_latent_val_unr_all, axis=0)

        self.assertAllClose(mpe_latent_val_dyn[-1], mpe_latent_val_unr)
        for do, uo in zip(mpe_ivs_val_dyn, mpe_ivs_val_unr):
            self.assertAllClose(np.squeeze(do), np.squeeze(uo))

    @parameterized.expand(arg_product(
        [True, False], [spn.InferenceType.MPE, spn.InferenceType.MARGINAL], [False, True]))
    def test_training(self, log, inf_type, iv_inputs):
        unrolled_root, var_nodes, unrolled_weights = get_dspn_unrolled(iv_inputs=iv_inputs)
        dynamic_root, dynamic_var_nodes, dynamic_weights = get_dspn(iv_inputs=iv_inputs)

        copy_weights = tf.group(*[tf.assign(dw.variable, uw.variable) for dw, uw in
                                  zip(dynamic_weights, unrolled_weights)])

        unrolled_feed, dynamic_feed = get_feed_dicts(var_nodes, dynamic_var_nodes,
                                                     iv_inputs)

        latent_dyn = dynamic_root.generate_ivs()
        latent_unr = unrolled_root.generate_ivs()

        accum_upd_dyn, learning_dyn, reset_acc_dyn, update_spn_dyn = self.setup_training(
            dynamic_root, inf_type, log=log)
        likelihood_dyn = learning_dyn.likelihood()

        accum_upd_unr, learning_unr, reset_acc_unr, update_spn_unr = self.setup_training(
            unrolled_root, inf_type, log=log)
        likelihood_unr = learning_unr.likelihood()

        latent_feed = np.random.randint(0, 2, size=BATCH_SIZE).reshape((BATCH_SIZE, 1))
        unrolled_feed[latent_unr] = latent_feed
        dynamic_feed[latent_dyn] = latent_feed

        init_dynamic = spn.initialize_weights(dynamic_root)
        init_unrolled = spn.initialize_weights(unrolled_root)

        with self.test_session() as sess:
            sess.run([init_dynamic, init_unrolled])
            sess.run(copy_weights)
            sess.run([reset_acc_dyn, reset_acc_unr])

            for _ in range(10):

                likelihood_dyn_val, _ = sess.run([likelihood_dyn, accum_upd_dyn],
                                                 feed_dict=dynamic_feed)
                sess.run(update_spn_dyn)

                likelihood_unr_val, _ = sess.run([likelihood_unr, accum_upd_unr],
                                                 feed_dict=unrolled_feed)
                sess.run(update_spn_unr)

                unrolled_weights_val = sess.run([w.variable for w in unrolled_weights])
                dynamic_weights_val = sess.run([w.variable for w in dynamic_weights])

                sess.run([reset_acc_dyn, reset_acc_unr])
                self.assertAllClose(likelihood_dyn_val, likelihood_unr_val)
                for wu, wd in zip(unrolled_weights_val, dynamic_weights_val):
                    self.assertAllClose(wu, wd)

    def setup_training(self, dynamic_root, inf_type, log):
        additive_smoothing_var = tf.constant(1.0, dtype=spn.conf.dtype)
        learning_dyn = spn.EMLearning(
            dynamic_root, value_inference_type=inf_type, additive_smoothing=additive_smoothing_var,
            log=log
        )
        reset_acc_dyn = learning_dyn.reset_accumulators()
        accum_upd_dyn = learning_dyn.accumulate_updates()
        update_spn_dyn = learning_dyn.update_spn()
        return accum_upd_dyn, learning_dyn, reset_acc_dyn, update_spn_dyn



