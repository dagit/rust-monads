use ghost::phantom;
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
    let c = call_cc(|exit1| {
        Cont::pure(1)
            .bind(move |_: i32| exit1('a'))
            .bind(|_: i32| unimplemented!())
    });
    println!("c = {:#?}", c.eval_cont());
    // We can also do a nested exit
    let c = call_cc(|exit1| -> Cont<char, char> {
        let exit1_clone = exit1.clone();
        call_cc(move |_: Rc<dyn Fn(char) -> Cont<char, char>>| exit1('c'))
            .bind(move |_: char| exit1_clone('b'))
    });
    println!("c = {:#?}", c.eval_cont());
}

#[derive(Clone)]
pub struct Cont<R, A> {
    //run: (a -> r) -> r
    pub run: Rc<dyn Fn(Rc<dyn Fn(A) -> R>) -> R>,
}

impl<R, A> Cont<R, A> {
    ///cont :: ((a -> r) -> r) -> Cont r a
    pub fn cont<F>(f: F) -> Cont<R, A>
    where
        F: Fn(Rc<dyn Fn(A) -> R>) -> R + 'static,
    {
        Cont { run: Rc::new(f) }
    }
    pub fn fmap<B, F>(self, f: F) -> Cont<R, B>
    where
        F: Fn(A) -> B + Clone + 'static,
        A: 'static,
        B: 'static,
        R: 'static,
    {
        //fmap f m = Cont $ \c -> run (c . f)
        //         = Cont (\c -> run (c . f))
        //         = Cont (\c -> run (\a -> c (f a)))
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(B) -> R>| {
                let f = f.clone();
                (self.run)(Rc::new(move |a: A| c(f(a))))
            }),
        }
    }
    pub fn pure(a: A) -> Cont<R, A>
    where
        A: 'static + Clone,
    {
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(A) -> R>| c(a.clone())),
        }
    }
    pub fn bind<B, K>(self, k: K) -> Cont<R, B>
    where
        K: Fn(A) -> Cont<R, B> + Clone + 'static,
        A: 'static,
        B: 'static,
        R: 'static,
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
        F: Fn(A) -> R + 'static,
    {
        (self.run)(Rc::new(move |a: A| f(a)))
    }
}
impl<R> Cont<R, R> {
    pub fn eval_cont(self) -> R {
        self.run_cont(|r: R| r)
    }
}

pub fn call_cc<A, B, R, F>(f: F) -> Cont<R, A>
where
    F: Fn(Rc<dyn Fn(A) -> Cont<R, B>>) -> Cont<R, A> + 'static,
    R: Clone,
    A: Clone + 'static,
    R: 'static,
{
    /* callCC f = ContT $ \ c -> runContT (f (\ x -> ContT $ \ _ -> c x)) c */
    let runit = call_cc_outer(f);
    Cont { run: runit }
}

fn call_cc_outer<A, B, R, F>(f: F) -> Rc<dyn Fn(Rc<dyn Fn(A) -> R>) -> R>
where
    F: Fn(Rc<dyn Fn(A) -> Cont<R, B>>) -> Cont<R, A> + 'static,
    A: Clone + 'static,
    R: 'static,
{
    Rc::new(move |c| {
        let inner = call_cc_inner(c.clone());
        (f(inner).run)(c)
    })
}

fn call_cc_inner<A, B, R>(c: Rc<dyn Fn(A) -> R>) -> Rc<dyn Fn(A) -> Cont<R, B>>
where
    A: Clone + 'static,
    R: 'static,
{
    Rc::new(move |a: A| {
        let c = c.clone();
        Cont {
            run: Rc::new(move |_| c(a.clone())),
        }
    })
}

// http://blog.sigfpe.com/2008/12/mother-of-all-monads.html
// i x = cont (\fred -> x >>= fred)
// where cont f = Cont { run: f }
// i :: Monad m => m a -> Cont (m b) a
pub fn lift<A, B, M, F>(m: This<M, A>) -> Cont<This<M, B>, A>
where
    M: Monad<A, B> + Clone,
    This<M, A>: Clone + 'static,
    A: Clone + 'static,
    B: Clone,
    This<M, B>: 'static,
{
    Cont::cont(move |f| {
        let m = m.clone();
        M::bind(m, move |a: A| f(a))
    })
}

// Heavily borrowed from here: https://github.com/RustyYato/type-families
pub type This<T, A> = <T as Family<A>>::This;
pub trait Family<A> {
    type This;
}

pub trait Functor<A, B>: Family<A> + Family<B> {
    fn map<F>(f: F, this: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + Clone + 'static;
}

pub trait Pure<A>: Family<A> {
    fn pure(value: A) -> This<Self, A>
    where
        A: Clone;
}

pub trait Applicative<A, B>: Functor<A, B> + Pure<A> + Pure<B> {
    fn apply<F>(a: This<Self, F>, b: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + Clone,
        Self: Applicative<F, A>,
        A: Clone,
        B: Clone + 'static,
    {
        Self::lift_a2(move |q: F, r| q(r), a, b)
    }

    fn lift_a2<C, F>(f: F, a: This<Self, A>, b: This<Self, B>) -> This<Self, C>
    where
        F: Fn(A, B) -> C + Clone + 'static,
        Self: Pure<C>,
        Self: Family<C>,
        A: Clone,
        B: Clone,
        C: Clone + 'static;
}

pub trait Monad<A, B>: Applicative<A, B> {
    fn bind<F>(a: This<Self, A>, f: F) -> This<Self, B>
    where
        F: Fn(A) -> This<Self, B> + Clone + 'static;
    fn compose<F, G, C>(f: F, g: G, a: A) -> This<Self, C>
    where
        F: FnOnce(A) -> This<Self, B> + 'static,
        G: Fn(B) -> This<Self, C> + Clone + 'static,
        Self: Monad<B, C>,
        This<Self, C>: 'static,
    {
        Self::bind(f(a), g)
    }
}

#[phantom]
#[derive(Clone, Copy)]
pub struct ContFamily<R>;

impl<R, A> Family<A> for ContFamily<R> {
    type This = Cont<R, A>;
}

impl<R, A: 'static> Pure<A> for ContFamily<R> {
    fn pure(value: A) -> This<Self, A>
    where
        A: Clone + 'static,
    {
        Cont::pure(value)
    }
}

impl<R: 'static, A: 'static, B: 'static> Functor<A, B> for ContFamily<R> {
    fn map<F>(f: F, this: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + Clone + 'static,
    {
        this.fmap(f)
    }
}

impl<R: Clone + 'static, A: Clone + 'static, B: Clone + 'static> Applicative<A, B>
    for ContFamily<R>
{
    fn lift_a2<C, F>(f: F, a: This<Self, A>, b: This<Self, B>) -> This<Self, C>
    where
        F: Fn(A, B) -> C + Clone + 'static,
        C: Clone + 'static,
    {
        a.bind(move |x: A| {
            let f = f.clone();
            let b = b.clone();
            b.bind(move |y: B| {
                let x = x.clone();
                Cont::pure(f(x, y))
            })
        })
    }
}

impl<A: 'static, B: 'static, R: 'static> Monad<A, B> for ContFamily<R>
where
    R: Clone,
    A: Clone,
    B: Clone,
{
    fn bind<K>(a: This<Self, A>, k: K) -> This<Self, B>
    where
        K: Fn(A) -> Cont<R, B> + Clone + 'static,
    {
        Cont::bind(a, k)
    }
}
