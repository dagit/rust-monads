use std::rc::Rc;

fn main() {
    println!("Hello, world!");
    let a = Cont::<i32, i32>::pure(1);
    println!("a = {}", a.clone().eval_cont());
    let b = a.fmap(|x| x + 1);
    println!("b = {}", b.clone().run_cont(|x| x));
    let c = b.bind(|x: i32| Cont::pure(x + x));
    println!("c = {}", c.run_cont(|x| x));
    let a = Cont::<i32, i32>::pure(1);
    let b = a.fmap(|x| x + 1);
    println!("b = {}", b.clone().eval_cont());
    let a = Cont::<i32, i32>::pure(1);
    let b = a.bind(|x: i32| Cont::pure(x + x));
    println!("b = {}", b.eval_cont());
    fn foo<'two, 'one: 'two>() {
        let c = call_cc(|exit1| -> Cont<'_, char, char> {
            let p = Cont::pure(1);
            let e = p.bind(move |_: i32| exit1('a'));
            let d: Cont<'_, char, char> = e.clone().bind(|_: char| unimplemented!());
            d
        });
        println!("c = {:#?}", c.eval_cont());
    }
    foo()
}

#[derive(Clone)]
pub struct Cont<'a, R, A> {
    //run: (a -> r) -> r
    pub run: Rc<dyn Fn(Rc<dyn Fn(A) -> R + 'a>) -> R + 'a>,
}

impl<'a, R, A> Cont<'a, R, A> {
    pub fn fmap<B: 'a, F>(self, f: F) -> Cont<'a, R, B>
    where
        F: Fn(A) -> B + 'a + Clone,
        A: 'a,
        R: 'a,
    {
        //fmap f m = Cont $ \c -> run (c . f)
        //         = Cont (\c -> run (c . f))
        //         = Cont (\c -> run (\a -> c (f a)))
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(B) -> R + 'a>| {
                let f = f.clone();
                (self.run)(Rc::new(move |a: A| c(f(a))))
            }),
        }
    }
    pub fn pure(a: A) -> Cont<'a, R, A>
    where
        A: 'a + Clone,
    {
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(A) -> R>| c(a.clone())),
        }
    }
    pub fn bind<B, K>(self, k: K) -> Cont<'a, R, B>
    where
        K: Fn(A) -> Cont<'a, R, B> + 'a + Clone,
        A: 'a,
        R: 'a,
        B: 'a,
    {
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(B) -> R>| {
                let k = k.clone();
                (self.run)(Rc::new(move |a: A| (k(a).run)(c.clone())))
            }),
        }
    }
    pub fn run_cont<F>(self, f: F) -> R
    where
        F: Fn(A) -> R + 'a,
    {
        (self.run)(Rc::new(move |a: A| f(a)))
    }
}
impl<'a, R> Cont<'a, R, R> {
    pub fn eval_cont(self) -> R {
        self.run_cont(|r: R| r)
    }
}

pub fn call_cc<'a, A, B, R, F>(f: F) -> Cont<'a, R, A>
where
    F: for<'b> Fn(Rc<dyn Fn(A) -> Cont<'a, R, B> + 'a>) -> Cont<'a, R, A> + 'a,
    R: 'a + Clone,
    A: 'a + Clone,
{
    /* callCC f = ContT $ \ c -> runContT (f (\ x -> ContT $ \ _ -> c x)) c */
    let runit = call_cc_outer(f);
    Cont { run: runit }
}

fn call_cc_outer<'a, A, B, R, F>(f: F) -> Rc<dyn for<'b> Fn(Rc<dyn Fn(A) -> R + 'a>) -> R + 'a>
where
    F: for<'b> Fn(Rc<dyn Fn(A) -> Cont<'a, R, B> + 'a>) -> Cont<'a, R, A> + 'a,
    A: Clone + 'a,
    R: 'a,
{
    Rc::new(move |c| {
        let inner = call_cc_inner(c.clone());
        (f(inner).run)(c)
    })
}

fn call_cc_inner<'a, A, B, R>(c: Rc<dyn Fn(A) -> R + 'a>) -> Rc<dyn Fn(A) -> Cont<'a, R, B> + 'a>
where
    A: 'a + Clone,
    R: 'a,
{
    Rc::new(move |a: A| {
        let c = c.clone();
        Cont {
            run: Rc::new(move |_| c(a.clone())),
        }
    })
}
